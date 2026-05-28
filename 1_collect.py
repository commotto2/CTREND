"""
1_collect.py — 데이터 수집 (Binance OHLCV + CoinGecko 시가총액)
================================================================
- Binance API : OHLCV (키 불필요)
- CoinGecko Demo API : 시가총액 (Demo 키 필요)

두 소스를 합쳐서 daily_prices에 저장.
시가총액이 없는 코인은 volume × close 로 대체.
"""

import os, sqlite3, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────
DB_PATH         = Path("data/prices.db")
BINANCE_URL     = "https://data-api.binance.vision/api/v3"
COINGECKO_URL   = "https://api.coingecko.com/api/v3"
CG_API_KEY      = os.environ.get("COINGECKO_API_KEY", "")

TARGET_COINS    = 300
FETCH_FROM      = "2020-01-01"
FETCH_FROM_MS   = int(datetime.strptime(FETCH_FROM, "%Y-%m-%d").timestamp() * 1000)

BINANCE_SLEEP   = 0.3
CG_SLEEP        = 2.5
CG_BACKOFF      = [60, 120]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── DB 초기화 ────────────────────────────────────────────
def init_db(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS coin_info (
        coin_id     TEXT PRIMARY KEY,
        symbol      TEXT,
        name        TEXT,
        rank        INTEGER,
        cg_id       TEXT,
        updated     TEXT
    );
    CREATE TABLE IF NOT EXISTS daily_prices (
        coin_id     TEXT,
        date        TEXT,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        volume      REAL,
        mcap        REAL,
        PRIMARY KEY (coin_id, date)
    );
    CREATE TABLE IF NOT EXISTS collection_status (
        coin_id     TEXT PRIMARY KEY,
        status      TEXT,
        mcap_status TEXT DEFAULT 'pending',
        attempt     INTEGER DEFAULT 0,
        updated     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_dp_coin ON daily_prices(coin_id);
    CREATE INDEX IF NOT EXISTS idx_dp_date ON daily_prices(date);
    """)
    conn.commit()
    log("DB 초기화 완료")

# ── Binance API ───────────────────────────────────────────
def binance_get(endpoint, params=None):
    url = BINANCE_URL + endpoint
    for attempt in range(4):
        try:
            time.sleep(BINANCE_SLEEP)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                log("  Binance 429 → 60초 대기")
                time.sleep(60)
            else:
                log(f"  Binance HTTP {r.status_code} (시도 {attempt+1})")
                time.sleep(10)
        except requests.RequestException as e:
            log(f"  Binance 네트워크 오류: {e}")
            time.sleep(15)
    return None

# ── CoinGecko API ─────────────────────────────────────────
def cg_get(endpoint, params=None):
    url = COINGECKO_URL + endpoint
    if params is None:
        params = {}
    if CG_API_KEY:
        params["x_cg_demo_api_key"] = CG_API_KEY
    backoff_idx = 0
    for attempt in range(4):
        try:
            time.sleep(CG_SLEEP)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = CG_BACKOFF[min(backoff_idx, len(CG_BACKOFF)-1)]
                log(f"  CoinGecko 429 → {wait}초 대기")
                time.sleep(wait)
                backoff_idx += 1
            elif r.status_code in (400, 404):
                return None
            else:
                log(f"  CoinGecko HTTP {r.status_code} (시도 {attempt+1})")
                time.sleep(10)
        except requests.RequestException as e:
            log(f"  CoinGecko 네트워크 오류: {e}")
            time.sleep(15)
    return None

# ── 코인 목록 수집 ────────────────────────────────────────
def fetch_coin_list(conn):
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    if n >= TARGET_COINS:
        log(f"코인 목록 이미 있음 ({n}개) → 스킵")
        return

    log("거래량 상위 코인 목록 수집 중 (Binance)...")
    data = binance_get("/ticker/24hr")
    if not data:
        raise RuntimeError("코인 목록 수집 실패")

    usdt = [d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 0]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    top = usdt[:TARGET_COINS]

    # CoinGecko ID 매핑 (시가총액 수집에 필요)
    log("CoinGecko ID 매핑 중...")
    cg_map = {}
    if CG_API_KEY:
        cg_data = cg_get("/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1
        })
        if cg_data:
            for c in cg_data:
                sym = c["symbol"].upper() + "USDT"
                cg_map[sym] = c["id"]
        time.sleep(CG_SLEEP)
        cg_data2 = cg_get("/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 2
        })
        if cg_data2:
            for c in cg_data2:
                sym = c["symbol"].upper() + "USDT"
                cg_map[sym] = c["id"]

    now = datetime.now().isoformat()
    cur.executemany(
        "INSERT OR IGNORE INTO coin_info VALUES(?,?,?,?,?,?)",
        [(d["symbol"],
          d["symbol"].replace("USDT",""),
          d["symbol"].replace("USDT",""),
          i+1,
          cg_map.get(d["symbol"]),
          now)
         for i, d in enumerate(top)]
    )
    cur.executemany(
        "INSERT OR IGNORE INTO collection_status(coin_id,status,mcap_status) VALUES(?,'pending','pending')",
        [(d["symbol"],) for d in top]
    )
    conn.commit()
    mapped = sum(1 for d in top if cg_map.get(d["symbol"]))
    log(f"코인 목록 저장: {len(top)}개 (CoinGecko 매핑: {mapped}개)")

# ── Binance OHLCV 수집 ────────────────────────────────────
def fetch_ohlcv(conn, symbol):
    all_rows = []
    start_ms = FETCH_FROM_MS

    while True:
        data = binance_get("/klines", {
            "symbol":    symbol,
            "interval":  "1d",
            "startTime": start_ms,
            "limit":     1000
        })
        if not data or len(data) == 0:
            break

        for k in data:
            date = datetime.utcfromtimestamp(k[0]/1000).strftime("%Y-%m-%d")
            all_rows.append((
                symbol, date,
                float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                float(k[5]), None
            ))

        if len(data) < 1000:
            break
        start_ms = data[-1][0] + 86400000

    if not all_rows:
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO daily_prices VALUES(?,?,?,?,?,?,?,?)",
        all_rows
    )
    conn.commit()
    return len(all_rows)

# ── CoinGecko 시가총액 수집 및 업데이트 ──────────────────
def fetch_mcap(conn, symbol, cg_id):
    """CoinGecko에서 시가총액 받아서 daily_prices.mcap 업데이트"""
    if not cg_id or not CG_API_KEY:
        return 0

    # 최근 365일 시가총액
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=364)).timestamp())

    data = cg_get(f"/coins/{cg_id}/market_chart/range", {
        "vs_currency": "usd",
        "from": start_ts,
        "to":   end_ts
    })
    if not data or "market_caps" not in data:
        return 0

    mcaps = {
        datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
        for ts, v in data["market_caps"]
        if v and v > 0
    }
    if not mcaps:
        return 0

    cur = conn.cursor()
    updated = 0
    for date, mcap in mcaps.items():
        result = cur.execute("""
            UPDATE daily_prices SET mcap=?
            WHERE coin_id=? AND date=? AND mcap IS NULL
        """, (mcap, symbol, date))
        updated += result.rowcount

    conn.commit()
    return updated

# ── mcap 없는 행 volume×close로 대체 ─────────────────────
def fill_mcap_fallback(conn):
    """시가총액 수집 못한 날짜는 volume × close로 채우기"""
    cur = conn.cursor()
    result = cur.execute("""
        UPDATE daily_prices
        SET mcap = volume * close
        WHERE mcap IS NULL AND volume IS NOT NULL AND close IS NOT NULL
    """)
    conn.commit()
    log(f"시가총액 대체값 적용: {result.rowcount:,}행 (volume × close)")

# ── 수집 현황 출력 ────────────────────────────────────────
def print_status(conn):
    cur = conn.cursor()
    total  = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    done   = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='done'").fetchone()[0]
    fail   = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='failed'").fetchone()[0]
    rows   = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    w_mcap = cur.execute("SELECT COUNT(*) FROM daily_prices WHERE mcap IS NOT NULL").fetchone()[0]
    pct    = w_mcap / rows * 100 if rows > 0 else 0
    log("=" * 50)
    log(f"코인: {total} | 완료: {done} | 실패: {fail}")
    log(f"daily_prices: {rows:,}행")
    log(f"시가총액 있음: {w_mcap:,}행 ({pct:.1f}%)")
    log("=" * 50)

# ── 메인 ─────────────────────────────────────────────────
def main():
    if not CG_API_KEY:
        log("⚠️  COINGECKO_API_KEY 없음 — 시가총액은 volume×close로 대체")

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    init_db(conn)
    fetch_coin_list(conn)

    cur = conn.cursor()

    # ── OHLCV 수집 ──────────────────────────────────────
    ohlcv_targets = cur.execute("""
        SELECT coin_id FROM collection_status
        WHERE status = 'pending'
        ORDER BY coin_id
    """).fetchall()

    log(f"OHLCV 수집 대상: {len(ohlcv_targets)}개")
    ok = fail = 0
    for (coin_id,) in ohlcv_targets:
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
                fail += 1
                log(f"  ✗ {coin_id}: failed")
            else:
                log(f"  ✗ {coin_id}: 재시도 예정")
                fail += 1
        conn.commit()

    if ok > 0:
        log(f"OHLCV 완료: 성공={ok}, 실패={fail}")

    # ── 시가총액 수집 (CoinGecko) ────────────────────────
    if CG_API_KEY:
        mcap_targets = cur.execute("""
            SELECT ci.coin_id, ci.cg_id
            FROM coin_info ci
            JOIN collection_status cs ON ci.coin_id = cs.coin_id
            WHERE cs.status = 'done'
            AND ci.cg_id IS NOT NULL
            AND cs.mcap_status = 'pending'
            ORDER BY ci.rank
        """).fetchall()

        log(f"시가총액 수집 대상: {len(mcap_targets)}개")
        for (coin_id, cg_id) in mcap_targets:
            updated = fetch_mcap(conn, coin_id, cg_id)
            cur.execute(
                "UPDATE collection_status SET mcap_status='done' WHERE coin_id=?",
                (coin_id,)
            )
            conn.commit()
            log(f"  ✓ mcap {coin_id}: {updated}행 업데이트")

    # 시가총액 없는 행 fallback 처리
    fill_mcap_fallback(conn)
    print_status(conn)
    conn.close()

if __name__ == "__main__":
    main()
