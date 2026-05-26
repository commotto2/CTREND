"""
CTREND Database Builder  v2.1  (GitHub Actions용)
===================================================
논문: "A Trend Factor for the Cross-Section of Cryptocurrency Returns"
      Fieberg et al. — JFQA Vol.60 No.7, 2025

[실행 환경]
  - GitHub Actions (ubuntu-latest) 에서 자동 실행
  - DB 파일(crypto_db.sqlite)을 레포 루트에 저장
  - Actions가 수집 완료 후 자동으로 commit & push

[수집 전략]
  - 1회 실행당 최대 DAILY_TARGET개 수집 (기본 250개)
  - 429 rate limit 시 exponential backoff 자동 적용
  - 중단 후 재실행 시 정확히 이어서 진행 (collection_status 테이블)
"""

import sqlite3, time, requests, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ── 설정 ────────────────────────────────────────────
DB_PATH        = Path("crypto_db.sqlite")
LOG_PATH       = Path("collection_log.txt")

FETCH_START    = "2014-01-01"
FETCH_END      = "2022-05-31"
ANALYSIS_START = "2015-04-01"

MIN_MCAP_USD   = 1_000_000
WINSOR_LOW     = 0.005
WINSOR_HIGH    = 0.995

TARGET_COINS   = 2000
COINS_PER_PAGE = 250
DAILY_TARGET   = 250      # 1회 실행당 수집 코인 수

INITIAL_SLEEP  = 3.0
MIN_SLEEP      = 2.0
MAX_SLEEP      = 8.0
BACKOFF_429    = [70, 120, 180]
MAX_RETRIES    = 3

BASE_URL       = "https://api.coingecko.com/api/v3"
FETCH_START_TS = int(datetime.strptime(FETCH_START, "%Y-%m-%d").timestamp())
FETCH_END_TS   = int(datetime.strptime(FETCH_END,   "%Y-%m-%d").timestamp())

# ── 로깅 ────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)          # Actions 로그에 즉시 출력
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ── DB 초기화 ────────────────────────────────────────
def init_db(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS coin_info (
        coin_id TEXT PRIMARY KEY, symbol TEXT, name TEXT,
        rank_at_collection INTEGER, added_at TEXT
    );
    CREATE TABLE IF NOT EXISTS daily_prices (
        coin_id TEXT, date TEXT, close REAL, volume REAL, mcap REAL,
        PRIMARY KEY (coin_id, date)
    );
    CREATE TABLE IF NOT EXISTS weekly_data (
        coin_id TEXT, week_end TEXT, close REAL, ret REAL, ret_w REAL,
        volume REAL, mcap REAL, n_days INTEGER,
        PRIMARY KEY (coin_id, week_end)
    );
    CREATE TABLE IF NOT EXISTS collection_status (
        coin_id TEXT PRIMARY KEY, status TEXT,
        attempt INTEGER DEFAULT 0, last_try TEXT, rows_saved INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS failed_coins (
        coin_id TEXT PRIMARY KEY, reason TEXT, failed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS idx_daily_coin ON daily_prices(coin_id);
    CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_prices(date);
    CREATE INDEX IF NOT EXISTS idx_weekly_coin ON weekly_data(coin_id);
    """)
    conn.execute("""INSERT OR IGNORE INTO meta(key,value) VALUES
        ('created_at',?),('fetch_start',?),('fetch_end',?),('version','2.1')""",
        (datetime.now().isoformat(), FETCH_START, FETCH_END))
    conn.commit()

# ── Rate Limiter ─────────────────────────────────────
class RateLimiter:
    def __init__(self):
        self.sleep_sec = INITIAL_SLEEP
        self.streak = 0
        self.backoff_idx = 0

    def wait(self): time.sleep(self.sleep_sec)

    def on_success(self):
        self.streak += 1
        self.backoff_idx = 0
        if self.streak >= 10 and self.sleep_sec > MIN_SLEEP:
            self.sleep_sec = max(MIN_SLEEP, self.sleep_sec - 0.2)
            self.streak = 0

    def on_429(self):
        wait = BACKOFF_429[min(self.backoff_idx, len(BACKOFF_429)-1)]
        log(f"  429 rate limit → {wait}초 대기")
        time.sleep(wait)
        self.sleep_sec = min(MAX_SLEEP, self.sleep_sec + 1.0)
        self.backoff_idx += 1
        self.streak = 0

rl = RateLimiter()

def api_get(endpoint, params=None):
    url = BASE_URL + endpoint
    for attempt in range(MAX_RETRIES + 1):
        try:
            rl.wait()
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                rl.on_success(); return r.json()
            elif r.status_code == 429:
                rl.on_429()
            elif r.status_code in (400, 404):
                return None
            else:
                log(f"  HTTP {r.status_code} — 재시도 {attempt+1}/{MAX_RETRIES}")
                time.sleep(10)
        except requests.RequestException as e:
            log(f"  네트워크 오류: {e}")
            time.sleep(15)
    return None

# ── 코인 목록 수집 ───────────────────────────────────
def fetch_coin_list(conn):
    cur = conn.cursor()
    existing = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    if existing >= TARGET_COINS:
        log(f"코인 목록 이미 존재 ({existing}개) → 스킵"); return

    log(f"코인 목록 수집 중 (상위 {TARGET_COINS}개)...")
    all_coins, pages = [], (TARGET_COINS + COINS_PER_PAGE - 1) // COINS_PER_PAGE
    for page in range(1, pages + 1):
        data = api_get("/coins/markets", {
            "vs_currency":"usd","order":"market_cap_desc",
            "per_page":COINS_PER_PAGE,"page":page,"sparkline":"false"
        })
        if not data: log(f"  페이지 {page} 실패"); break
        all_coins.extend(data)
        log(f"  {page}/{pages} 완료 ({len(all_coins)}개)")

    now = datetime.now().isoformat()
    cur.executemany(
        "INSERT OR IGNORE INTO coin_info VALUES(?,?,?,?,?)",
        [(c["id"],c["symbol"],c["name"],i+1,now) for i,c in enumerate(all_coins)]
    )
    cur.executemany(
        "INSERT OR IGNORE INTO collection_status(coin_id,status) VALUES(?,'pending')",
        [(c["id"],) for c in all_coins]
    )
    conn.commit()
    log(f"코인 목록 저장 완료: {len(all_coins)}개")

# ── 코인 1개 가격 수집 ───────────────────────────────
def fetch_coin_prices(conn, coin_id):
    data = api_get(f"/coins/{coin_id}/market_chart/range",
        {"vs_currency":"usd","from":FETCH_START_TS,"to":FETCH_END_TS})
    if not data or "prices" not in data: return -1

    prices  = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts,v in data.get("prices",[])}
    volumes = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts,v in data.get("total_volumes",[])}
    mcaps   = {datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"): v
               for ts,v in data.get("market_caps",[])}

    rows = [(coin_id, d, prices[d], volumes.get(d), mcaps.get(d))
            for d in sorted(prices) if prices[d] and prices[d] > 0]
    if not rows: return 0

    conn.executemany(
        "INSERT OR REPLACE INTO daily_prices VALUES(?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)

# ── 주간 집계 ────────────────────────────────────────
def compute_weekly(conn):
    log("주간 집계 중...")
    df = pd.read_sql("""SELECT coin_id,date,close,volume,mcap FROM daily_prices
        WHERE date>=? AND date<=? ORDER BY coin_id,date""",
        conn, params=(FETCH_START, FETCH_END))
    if df.empty: return

    df["date"] = pd.to_datetime(df["date"])
    df["week_end"] = df["date"].dt.to_period("W-FRI").apply(lambda p: p.end_time.date())
    grp = df.groupby(["coin_id","week_end"])
    weekly = pd.DataFrame({
        "close": grp["close"].last(), "volume": grp["volume"].sum(),
        "mcap":  grp["mcap"].last(),  "n_days": grp["close"].count()
    }).reset_index()
    weekly = weekly.sort_values(["coin_id","week_end"])
    weekly["ret"]   = weekly.groupby("coin_id")["close"].pct_change()
    lo, hi          = weekly["ret"].quantile(WINSOR_LOW), weekly["ret"].quantile(WINSOR_HIGH)
    weekly["ret_w"] = weekly["ret"].clip(lo, hi)
    weekly["week_end"] = weekly["week_end"].astype(str)

    conn.execute("DELETE FROM weekly_data")
    weekly[["coin_id","week_end","close","ret","ret_w","volume","mcap","n_days"]
           ].to_sql("weekly_data", conn, if_exists="append", index=False)
    conn.commit()
    log(f"주간 집계 완료: {len(weekly):,}행")

# ── 수집 현황 출력 ───────────────────────────────────
def print_status(conn):
    cur = conn.cursor()
    total   = cur.execute("SELECT COUNT(*) FROM coin_info").fetchone()[0]
    done    = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='done'").fetchone()[0]
    pending = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='pending'").fetchone()[0]
    failed  = cur.execute("SELECT COUNT(*) FROM collection_status WHERE status='failed'").fetchone()[0]
    dp      = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    wk      = cur.execute("SELECT COUNT(*) FROM weekly_data").fetchone()[0]

    pct = done/total*100 if total else 0
    log("=" * 50)
    log(f"코인 목록 : {total:,}개")
    log(f"수집 완료 : {done:,}개 ({pct:.1f}%)")
    log(f"수집 대기 : {pending:,}개")
    log(f"수집 실패 : {failed:,}개")
    log(f"daily_prices : {dp:,}행")
    log(f"weekly_data  : {wk:,}행")
    log("=" * 50)

    if pending == 0 and done > 0:
        log("✅ 전체 수집 완료 — Step 2 (기술 지표)로 진행 가능")
    else:
        sessions = (pending // DAILY_TARGET) + (1 if pending % DAILY_TARGET else 0)
        log(f"예상 남은 실행 횟수: {sessions}회")

# ── 메인 ─────────────────────────────────────────────
def main():
    log("CTREND DB Builder v2.1 시작")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    init_db(conn)
    fetch_coin_list(conn)

    cur = conn.cursor()
    pending = cur.execute(
        "SELECT COUNT(*) FROM collection_status WHERE status='pending'"
    ).fetchone()[0]

    if pending == 0:
        log("수집 완료 상태 — 주간 집계만 갱신")
        compute_weekly(conn)
    else:
        log(f"수집 시작 (이번 실행: 최대 {DAILY_TARGET}개)")
        success = fail = 0
        targets = cur.execute("""
            SELECT cs.coin_id FROM collection_status cs
            LEFT JOIN failed_coins fc ON cs.coin_id=fc.coin_id
            WHERE cs.status='pending' AND fc.coin_id IS NULL
            LIMIT ?""", (DAILY_TARGET,)).fetchall()

        for (coin_id,) in targets:
            now = datetime.now().isoformat()
            cur.execute("UPDATE collection_status SET attempt=attempt+1,last_try=? WHERE coin_id=?",
                        (now, coin_id))
            conn.commit()

            rows = fetch_coin_prices(conn, coin_id)
            if rows >= 0:
                cur.execute("UPDATE collection_status SET status='done',rows_saved=?,last_try=? WHERE coin_id=?",
                            (rows, now, coin_id))
                success += 1
                log(f"  ✓ {coin_id}: {rows}행")
            else:
                attempt = cur.execute(
                    "SELECT attempt FROM collection_status WHERE coin_id=?", (coin_id,)
                ).fetchone()[0]
                if attempt >= MAX_RETRIES:
                    cur.execute("UPDATE collection_status SET status='failed' WHERE coin_id=?", (coin_id,))
                    cur.execute("INSERT OR REPLACE INTO failed_coins VALUES(?,?,?)",
                                (coin_id, "max retries", now))
                    log(f"  ✗ {coin_id}: failed 처리")
                    fail += 1
                else:
                    log(f"  ✗ {coin_id}: 재시도 대기")
                    fail += 1
            conn.commit()

        log(f"이번 실행 결과: 성공={success}, 실패={fail}")
        compute_weekly(conn)

    print_status(conn)
    conn.close()
    log("완료")

if __name__ == "__main__":
    main()
