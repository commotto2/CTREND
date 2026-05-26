"""
1_collect.py — 데이터 수집
============================
CoinGecko Demo API로 상위 300개 코인의 일별 OHLCV + 시가총액 수집.
수집 결과를 data/prices.db (SQLite)에 저장.

- 매주 금요일 GitHub Actions가 자동 실행
- 중단/재실행 시 이어서 수집 (collection_status 테이블)
- 429 발생 시 자동 backoff
"""

import os, sqlite3, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────
API_KEY      = os.environ.get("COINGECKO_API_KEY", "")
BASE_URL     = "https://api.coingecko.com/api/v3"
DB_PATH      = Path("data/prices.db")
TARGET_COINS = 300
SLEEP_SEC    = 2.5          # 호출 간격 (Demo: 분당 100콜 → 여유있게 2.5초)
BACKOFF_429  = [60, 120]    # 429 시 대기(초)

# 수집 기간: 논문 종료(2022-06) 직전 ~ 현재
FETCH_FROM   = "2022-01-01"   # 약간 겹치게 잡아서 연속성 확보
FETCH_FROM_TS = int(datetime.strptime(FETCH_FROM, "%Y-%m-%d").timestamp())
FETCH_TO_TS   = int(datetime.now().timestamp())

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
        close     REAL,
        volume    REAL,
        mcap      REAL,
        PRIMARY KEY (coin_id, date)
    );
    CREATE TABLE IF NOT EXISTS collection_status (
        coin_id   TEXT PRIMARY KEY,
        status    TEXT,   -- pending / done / failed
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
    headers = {"x-cg-demo-api-key": API_KEY} if API_KEY else {}
    backoff_idx = 0

    for attempt in range(4):
        try:
            time.sleep(SLEEP_SEC)
            r = requests.get(url, params=params, headers=headers, timeout=30)

            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = BACKOFF_429[min(backoff_idx, len(BACKOFF_429)-1)]
                log(f"  429 → {wait}초 대기")
                time.sleep(wait)
                backoff_idx += 1
            elif r.status_code in (400, 404):
                return None
            else:
                log(f"  HTTP {r.status_code} (시도 {attempt+1})")
                time.sleep(15)
        except requests.RequestException as e:
            log(f"  네트워크 오류: {e}")
            time.sleep(15)
    return None

# ── 코인 목록 수집 ────────────────────────────────────────
def fetch_coin_list(conn):
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    if n >= TARGET_COINS:
        log(f"코인 목록 이미 있음 ({n}개) → 스킵")
        return

    log(f"코인 목록 수집 중 (상위 {TARGET_COINS}개)...")
    coins, page = [], 1
    while len(coins) < TARGET_COINS:
        batch = api_get("/coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": page, "sparkline": "false"
        })
        if not batch:
            break
        coins.extend(batch)
        log(f"  {len(coins)}개 수집")
        page += 1

    coins = coins[:TARGET_COINS]
    now = datetime.now().isoformat()
    cur.executemany(
        "INSERT OR IGNORE INTO coin_info VALUES(?,?,?,?,?)",
        [(c["id"], c["symbol"], c["name"], i+1, now) for i, c in enumerate(coins)]
    )
    cur.executemany(
        "INSERT OR IGNORE INTO collection_status(coin_id,status) VALUES(?,'pending')",
        [(c["id"],) for c in coins]
    )
    conn.commit()
    log(f"코인 목록 저장: {len(coins)}개")

# ── 코인 1개 가격 수집 ────────────────────────────────────
def fetch_prices(conn, coin_id):
    data = api_get(f"/coins/{coin_id}/market_chart/range", {
        "vs_currency": "usd",
        "from": FETCH_FROM_TS,
        "to":   FETCH_TO_TS
    })
    if not data or "prices" not in data:
        return -1

    prices  = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts, v in data.get("prices", [])}
    volumes = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts, v in data.get("total_volumes", [])}
    mcaps   = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts, v in data.get("market_caps", [])}

    rows = [(coin_id, d, prices[d], volumes.get(d), mcaps.get(d))
            for d in sorted(prices) if prices[d] and prices[d] > 0]
    if not rows:
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO daily_prices VALUES(?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)

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
    if not API_KEY:
        log("⚠️  COINGECKO_API_KEY 환경변수가 없습니다")
        log("   GitHub Secret에 COINGECKO_API_KEY를 설정하세요")
        raise SystemExit(1)

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
        cur.execute("UPDATE collection_status SET attempt=attempt+1, updated=? WHERE coin_id=?",
                    (now, coin_id))
        conn.commit()

        rows = fetch_prices(conn, coin_id)

        if rows >= 0:
            cur.execute("UPDATE collection_status SET status='done', updated=? WHERE coin_id=?",
                        (now, coin_id))
            ok += 1
            log(f"  ✓ {coin_id}: {rows}행")
        else:
            attempt = cur.execute(
                "SELECT attempt FROM collection_status WHERE coin_id=?", (coin_id,)
            ).fetchone()[0]
            if attempt >= 3:
                cur.execute("UPDATE collection_status SET status='failed', updated=? WHERE coin_id=?",
                            (now, coin_id))
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
