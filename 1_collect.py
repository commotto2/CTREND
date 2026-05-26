"""
1_collect.py — 데이터 수집 (Binance API)
==========================================
Binance 공개 API로 USDT 페어 코인의 일별 OHLCV 수집.
- API 키 불필요
- 2017년부터 현재까지 데이터 제공
- OHLCV 전체 제공 (Open/High/Low/Close/Volume)
- 시가총액 없음 → 거래량(volume)으로 대체

수집 결과: data/prices.db (SQLite)
"""

import sqlite3, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────
DB_PATH      = Path("data/prices.db")
BASE_URL     = "https://api.binance.com/api/v3"
TARGET_COINS = 300        # 거래량 상위 N개
FETCH_FROM   = "2022-01-01"   # 수집 시작일 (논문 종료 직전)
SLEEP_SEC    = 0.3            # 호출 간격 (Binance는 rate limit 넉넉함)

FETCH_FROM_MS = int(datetime.strptime(FETCH_FROM, "%Y-%m-%d").timestamp() * 1000)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── DB 초기화 ────────────────────────────────────────────
def init_db(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS coin_info (
        coin_id   TEXT PRIMARY KEY,
        symbol    TEXT,
        name      TEXT,
        rank      INTEGER,
        updated   TEXT
    );
    CREATE TABLE IF NOT EXISTS daily_prices (
        coin_id   TEXT,
        date      TEXT,
        open      REAL,
        high      REAL,
        low       REAL,
        close     REAL,
        volume    REAL,
        mcap      REAL,
        PRIMARY KEY (coin_id, date)
    );
    CREATE TABLE IF NOT EXISTS collection_status (
        coin_id   TEXT PRIMARY KEY,
        status    TEXT,
        attempt   INTEGER DEFAULT 0,
        updated   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dp_coin ON daily_prices(coin_id);
    CREATE INDEX IF NOT EXISTS idx_dp_date ON daily_prices(date);
    """)
    conn.commit()
    log("DB 초기화 완료")

# ── API 호출 ─────────────────────────────────────────────
def api_get(endpoint, params=None):
    url = BASE_URL + endpoint
    for attempt in range(4):
        try:
            time.sleep(SLEEP_SEC)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                log(f"  429 → 60초 대기")
                time.sleep(60)
            else:
                log(f"  HTTP {r.status_code} (시도 {attempt+1})")
                time.sleep(10)
        except requests.RequestException as e:
            log(f"  네트워크 오류: {e}")
            time.sleep(15)
    return None

# ── 거래량 상위 USDT 페어 목록 ────────────────────────────
def fetch_coin_list(conn):
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    if n >= TARGET_COINS:
        log(f"코인 목록 이미 있음 ({n}개) → 스킵")
        return

    log("거래량 상위 코인 목록 수집 중...")

    # 24시간 티커 (거래량 기준 정렬)
    data = api_get("/ticker/24hr")
    if not data:
        raise RuntimeError("코인 목록 수집 실패")

    # USDT 페어만 필터
    usdt = [d for d in data if d["symbol"].endswith("USDT")
            and float(d["quoteVolume"]) > 0]

    # 거래량 내림차순 정렬
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    top = usdt[:TARGET_COINS]

    now = datetime.now().isoformat()
    cur.executemany(
        "INSERT OR IGNORE INTO coin_info VALUES(?,?,?,?,?)",
        [(d["symbol"], d["symbol"].replace("USDT",""), d["symbol"].replace("USDT",""), i+1, now)
         for i, d in enumerate(top)]
    )
    cur.executemany(
        "INSERT OR IGNORE INTO collection_status(coin_id,status) VALUES(?,'pending')",
        [(d["symbol"],) for d in top]
    )
    conn.commit()
    log(f"코인 목록 저장: {len(top)}개")

# ── 코인 1개 OHLCV 수집 ──────────────────────────────────
def fetch_ohlcv(conn, symbol):
    """
    Binance klines API: 한 번에 최대 1000개 → 여러 번 나눠서 수집
    """
    all_rows = []
    start_ms = FETCH_FROM_MS

    while True:
        data = api_get("/klines", {
            "symbol":    symbol,
            "interval":  "1d",
            "startTime": start_ms,
            "limit":     1000
        })
        if not data:
            break
        if len(data) == 0:
            break

        for k in data:
            date   = datetime.utcfromtimestamp(k[0]/1000).strftime("%Y-%m-%d")
            o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            # mcap 없음 → None (거래량으로 대체)
            all_rows.append((symbol, date, o, h, l, c, v, None))

        # 마지막 캔들 다음부터 다시 요청
        last_ts = data[-1][0]
        start_ms = last_ts + 86400000  # +1일(ms)

        # 오늘까지 다 받았으면 종료
        if len(data) < 1000:
            break

    if not all_rows:
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO daily_prices VALUES(?,?,?,?,?,?,?,?)",
        all_rows
    )
    conn.commit()
    return len(all_rows)

# ── 수집 현황 출력 ────────────────────────────────────────
def print_status(conn):
    cur = conn.cursor()
    total   = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    done    = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='done'").fetchone()[0]
    pending = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='pending'").fetchone()[0]
    failed  = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='failed'").fetchone()[0]
    rows    = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    log("=" * 45)
    log(f"코인: {total} | 완료: {done} | 대기: {pending} | 실패: {failed}")
    log(f"daily_prices: {rows:,}행")
    if pending == 0:
        log("✅ 전체 수집 완료")
    log("=" * 45)

# ── 메인 ─────────────────────────────────────────────────
def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    init_db(conn)
    fetch_coin_list(conn)

    cur = conn.cursor()
    targets = cur.execute("""
        SELECT coin_id FROM collection_status
        WHERE status = 'pending'
        ORDER BY coin_id
    """).fetchall()

    log(f"수집 대상: {len(targets)}개")
    ok = fail = 0

    for (coin_id,) in targets:
        now = datetime.now().isoformat()
        cur.execute(
            "UPDATE collection_status SET attempt=attempt+1, updated=? WHERE coin_id=?",
            (now, coin_id)
        )
        conn.commit()

        rows = fetch_ohlcv(conn, coin_id)

        if rows >= 0:
            cur.execute(
                "UPDATE collection_status SET status='done', updated=? WHERE coin_id=?",
                (now, coin_id)
            )
            ok += 1
            log(f"  ✓ {coin_id}: {rows}행")
        else:
            attempt = cur.execute(
                "SELECT attempt FROM collection_status WHERE coin_id=?", (coin_id,)
            ).fetchone()[0]
            if attempt >= 3:
                cur.execute(
                    "UPDATE collection_status SET status='failed', updated=? WHERE coin_id=?",
                    (now, coin_id)
                )
                log(f"  ✗ {coin_id}: failed")
                fail += 1
            else:
                log(f"  ✗ {coin_id}: 재시도 예정")
                fail += 1
        conn.commit()

    log(f"완료: 성공={ok}, 실패={fail}")
    print_status(conn)
    conn.close()

if __name__ == "__main__":
    main()
