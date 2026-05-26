"""
2_indicators.py — 기술 지표 계산
==================================
Binance OHLCV 데이터로 논문의 28개 기술 지표 계산.
(Binance는 Open/High/Low/Close/Volume 전체 제공 → 논문과 동일 조건)

지표 목록:
  rsi, stoch_rsi, stoch_k, stoch_d, cci
  sma_3d ~ sma_200d (7개)
  ppo, ppo_diff
  vsma_3d ~ vsma_200d (7개)
  pvo, pvo_diff
  chaikin
  boll_low, boll_mid, boll_high, boll_width
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/prices.db")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 지표 함수 ─────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close, rsi_period=14, smooth=5):
    rsi      = calc_rsi(close, rsi_period)
    rsi_min  = rsi.rolling(rsi_period).min()
    rsi_max  = rsi.rolling(rsi_period).max()
    stoch    = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    return stoch.rolling(smooth).mean()

def calc_stochastic(close, high, low, k_period=14, d_smooth=3, slow_k=14):
    lo     = low.rolling(k_period).min()
    hi     = high.rolling(k_period).max()
    k      = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    k_slow = k.rolling(slow_k).mean()
    d      = k_slow.rolling(d_smooth).mean()
    return k_slow, d

def calc_cci(close, high, low, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_sma_ratio(close, period):
    return close.rolling(period).mean() / close

def calc_ppo(close, fast=12, slow=26, signal=9):
    ema_f  = close.ewm(span=fast, adjust=False).mean()
    ema_s  = close.ewm(span=slow, adjust=False).mean()
    ppo    = (ema_f - ema_s) / ema_s.replace(0, np.nan) * 100
    sig    = ppo.ewm(span=signal, adjust=False).mean()
    return ppo, ppo - sig

def calc_volume_sma_ratio(volume, period):
    vol = volume.replace(0, np.nan)
    return vol.rolling(period).mean() / vol

def calc_pvo(volume, fast=12, slow=26, signal=9):
    vol    = volume.replace(0, np.nan)
    ema_f  = vol.ewm(span=fast, adjust=False).mean()
    ema_s  = vol.ewm(span=slow, adjust=False).mean()
    pvo    = (ema_f - ema_s) / ema_s.replace(0, np.nan) * 100
    sig    = pvo.ewm(span=signal, adjust=False).mean()
    return pvo, pvo - sig

def calc_chaikin(close, high, low, volume, period=21, min_obs=10):
    hl  = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    return (mfv.rolling(period, min_periods=min_obs).sum() /
            volume.rolling(period, min_periods=min_obs).sum().replace(0, np.nan))

def calc_bollinger(close, period=20, std_dev=2):
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    b_hi  = ma + std_dev * std
    b_lo  = ma - std_dev * std
    return (b_lo / close,
            ma   / close,
            b_hi / close,
            (b_hi - b_lo) / ma.replace(0, np.nan))

# ── 코인 1개 전체 지표 계산 ───────────────────────────────

INDICATOR_COLS = [
    "rsi", "stoch_rsi", "stoch_k", "stoch_d", "cci",
    "sma_3d","sma_5d","sma_10d","sma_20d","sma_50d","sma_100d","sma_200d",
    "ppo","ppo_diff",
    "vsma_3d","vsma_5d","vsma_10d","vsma_20d","vsma_50d","vsma_100d","vsma_200d",
    "pvo","pvo_diff",
    "chaikin",
    "boll_low","boll_mid","boll_high","boll_width"
]

def calc_all_indicators(df):
    df     = df.sort_values("date").copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["rsi"]                        = calc_rsi(close)
    df["stoch_rsi"]                  = calc_stoch_rsi(close)
    df["stoch_k"], df["stoch_d"]     = calc_stochastic(close, high, low)
    df["cci"]                        = calc_cci(close, high, low)

    for n in [3, 5, 10, 20, 50, 100, 200]:
        df[f"sma_{n}d"] = calc_sma_ratio(close, n)

    df["ppo"], df["ppo_diff"]        = calc_ppo(close)

    for n in [3, 5, 10, 20, 50, 100, 200]:
        df[f"vsma_{n}d"] = calc_volume_sma_ratio(volume, n)

    df["pvo"], df["pvo_diff"]        = calc_pvo(volume)
    df["chaikin"]                    = calc_chaikin(close, high, low, volume)

    (df["boll_low"], df["boll_mid"],
     df["boll_high"], df["boll_width"]) = calc_bollinger(close)

    return df

# ── 주간 집계 (금요일 기준) ───────────────────────────────

def resample_weekly(df):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    cols = INDICATOR_COLS + ["open","high","low","close","volume"]
    weekly = df[cols].resample("W-FRI").last()
    weekly["ret"]  = weekly["close"].pct_change()
    # 거래량을 시가총액 대용으로 사용
    weekly["mcap"] = weekly["volume"] * weekly["close"]
    weekly["week"] = weekly.index.strftime("%G%V").astype(int)
    return weekly.reset_index()

# ── 메인 ─────────────────────────────────────────────────

def main():
    conn  = sqlite3.connect(DB_PATH)
    coins = pd.read_sql("""
        SELECT ci.coin_id FROM coin_info ci
        JOIN collection_status cs ON ci.coin_id = cs.coin_id
        WHERE cs.status = 'done'
    """, conn)["coin_id"].tolist()

    log(f"지표 계산 대상: {len(coins)}개 코인")

    conn.execute("DROP TABLE IF EXISTS weekly_indicators")
    conn.execute(f"""
        CREATE TABLE weekly_indicators (
            coin_id TEXT,
            date    TEXT,
            week    INTEGER,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            mcap    REAL,
            ret     REAL,
            {chr(10).join(f"{c} REAL," for c in INDICATOR_COLS[:-1])}
            {INDICATOR_COLS[-1]} REAL,
            PRIMARY KEY (coin_id, date)
        )
    """)
    conn.commit()

    all_weekly = []
    for i, coin_id in enumerate(coins):
        df = pd.read_sql(
            "SELECT date,open,high,low,close,volume FROM daily_prices WHERE coin_id=? ORDER BY date",
            conn, params=(coin_id,)
        )
        if len(df) < 30:
            continue
        try:
            df_ind  = calc_all_indicators(df)
            df_week = resample_weekly(df_ind)
            df_week["coin_id"] = coin_id
            all_weekly.append(df_week)
        except Exception as e:
            log(f"  ✗ {coin_id}: {e}")
            continue
        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(coins)} 완료")

    if all_weekly:
        result = pd.concat(all_weekly, ignore_index=True)
        cols   = ["coin_id","date","week","open","high","low","close","volume","mcap","ret"] + INDICATOR_COLS
        result["date"] = result["date"].astype(str)
        result[cols].to_sql("weekly_indicators", conn, if_exists="append", index=False)
        conn.commit()
        log(f"주간 지표 저장: {len(result):,}행 / {result['coin_id'].nunique()}개 코인")
    else:
        log("저장할 데이터 없음")

    conn.close()

if __name__ == "__main__":
    main()
