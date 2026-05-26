"""
2_indicators.py — 기술 지표 계산
==================================
논문의 28개 기술 지표를 일별 → 주간 집계 후 계산.

지표 목록 (논문 b02CalculateIndicators.m 기준):
  [가격 기반]
  - RSI (14일)
  - Stochastic RSI (14일, smooth 5)
  - Stochastic K/D (14일, smooth 3/14)
  - CCI (20일)
  - SMA 비율: 3/5/10/20/50/100/200일 (7개)
  - PPO (MACD 비율 버전) + Signal 차이

  [거래량 기반]
  - Volume SMA 비율: 3/5/10/20/50/100/200일 (7개)
  - PVO (Volume 오실레이터) + Signal 차이
  - Chaikin Money Flow (21일)
  - Bollinger Band Low/Mid/High/Width (4개)

총 28개
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/prices.db")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ════════════════════════════════════════════════════
# 개별 지표 함수
# ════════════════════════════════════════════════════

def calc_rsi(close, period=14):
    """RSI"""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close, rsi_period=14, smooth=5):
    """Stochastic RSI"""
    rsi      = calc_rsi(close, rsi_period)
    rsi_min  = rsi.rolling(rsi_period).min()
    rsi_max  = rsi.rolling(rsi_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    return stoch_rsi.rolling(smooth).mean()

def calc_stochastic(close, high, low, k_period=14, d_smooth=3, slow_k=14):
    """Stochastic Oscillator K/D
    close-only 모드: high/low가 없으면 close로 대체
    """
    h = high if high is not None else close
    l = low  if low  is not None else close
    lo = l.rolling(k_period).min()
    hi = h.rolling(k_period).max()
    k  = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    k_slow = k.rolling(slow_k).mean()
    d  = k_slow.rolling(d_smooth).mean()
    return k_slow, d

def calc_cci(close, high, low, period=20):
    """CCI — close-only 모드 지원"""
    h = high if high is not None else close
    l = low  if low  is not None else close
    tp = (h + l + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_sma_ratio(close, period):
    """SMA(n) / Close"""
    return close.rolling(period).mean() / close

def calc_ppo(close, fast=12, slow=26, signal=9):
    """PPO (Percentage Price Oscillator) + Signal 차이"""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    ppo        = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100
    ppo_signal = ppo.ewm(span=signal, adjust=False).mean()
    return ppo, ppo - ppo_signal

def calc_volume_sma_ratio(volume, period):
    """Volume SMA(n) / Volume"""
    vol = volume.replace(0, np.nan)
    return vol.rolling(period).mean() / vol

def calc_pvo(volume, fast=12, slow=26, signal=9):
    """PVO (Percentage Volume Oscillator) + Signal 차이"""
    vol        = volume.replace(0, np.nan)
    ema_fast   = vol.ewm(span=fast, adjust=False).mean()
    ema_slow   = vol.ewm(span=slow, adjust=False).mean()
    pvo        = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100
    pvo_signal = pvo.ewm(span=signal, adjust=False).mean()
    return pvo, pvo - pvo_signal

def calc_chaikin(close, high, low, volume, period=21, min_obs=10):
    """Chaikin Money Flow"""
    h   = high   if high   is not None else close
    l   = low    if low    is not None else close
    hl  = (h - l).replace(0, np.nan)
    mfm = ((close - l) - (h - close)) / hl
    mfv = mfm * volume
    return mfv.rolling(period, min_periods=min_obs).sum() / \
           volume.rolling(period, min_periods=min_obs).sum().replace(0, np.nan)

def calc_bollinger(close, period=20, std_dev=2):
    """Bollinger Bands: low/mid/high/width (모두 close 대비 비율)"""
    ma   = close.rolling(period).mean()
    std  = close.rolling(period).std()
    high = ma + std_dev * std
    low  = ma - std_dev * std
    boll_low   = low  / close
    boll_mid   = ma   / close
    boll_high  = high / close
    boll_width = (high - low) / ma.replace(0, np.nan)
    return boll_low, boll_mid, boll_high, boll_width

# ════════════════════════════════════════════════════
# 코인 1개 지표 계산
# ════════════════════════════════════════════════════

def calc_all_indicators(df):
    """
    df: coin별 일별 데이터 (date, close, volume, mcap)
    반환: 지표가 추가된 DataFrame (일별)
    """
    df = df.sort_values("date").copy()
    close  = df["close"]
    volume = df["volume"]
    # 이 데이터소스는 OHLCV 중 close/volume만 있음
    # high/low는 None으로 처리 → close-only 모드
    high = None
    low  = None

    # ── 가격 지표 ─────────────────────
    df["rsi"]           = calc_rsi(close)
    df["stoch_rsi"]     = calc_stoch_rsi(close)
    df["stoch_k"], df["stoch_d"] = calc_stochastic(close, high, low)
    df["cci"]           = calc_cci(close, high, low)

    for n in [3, 5, 10, 20, 50, 100, 200]:
        df[f"sma_{n}d"]  = calc_sma_ratio(close, n)

    df["ppo"], df["ppo_diff"] = calc_ppo(close)

    # ── 거래량 지표 ───────────────────
    for n in [3, 5, 10, 20, 50, 100, 200]:
        df[f"vsma_{n}d"] = calc_volume_sma_ratio(volume, n)

    df["pvo"], df["pvo_diff"] = calc_pvo(volume)
    df["chaikin"]       = calc_chaikin(close, high, low, volume)

    # ── Bollinger ─────────────────────
    df["boll_low"], df["boll_mid"], df["boll_high"], df["boll_width"] = \
        calc_bollinger(close)

    return df

# ════════════════════════════════════════════════════
# 주간 집계 (논문 방식: Liu et al. 기준 주 금요일)
# ════════════════════════════════════════════════════

INDICATOR_COLS = [
    "rsi", "stoch_rsi", "stoch_k", "stoch_d", "cci",
    "sma_3d", "sma_5d", "sma_10d", "sma_20d", "sma_50d", "sma_100d", "sma_200d",
    "ppo", "ppo_diff",
    "vsma_3d", "vsma_5d", "vsma_10d", "vsma_20d", "vsma_50d", "vsma_100d", "vsma_200d",
    "pvo", "pvo_diff",
    "chaikin",
    "boll_low", "boll_mid", "boll_high", "boll_width"
]

def resample_weekly(df):
    """일별 → 주간 (금요일 기준, 마지막 유효값 사용)"""
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # W-FRI: 금요일 기준 주
    weekly = df[INDICATOR_COLS + ["close", "volume", "mcap"]].resample("W-FRI").last()
    weekly["ret"] = weekly["close"].pct_change()
    weekly["week"] = weekly.index.strftime("%G%V").astype(int)  # YYYYWW
    return weekly.reset_index()

# ════════════════════════════════════════════════════
# 전체 실행
# ════════════════════════════════════════════════════

def main():
    conn = sqlite3.connect(DB_PATH)

    # 수집 완료된 코인만
    coins = pd.read_sql("""
        SELECT ci.coin_id FROM coin_info ci
        JOIN collection_status cs ON ci.coin_id = cs.coin_id
        WHERE cs.status = 'done'
    """, conn)["coin_id"].tolist()

    log(f"지표 계산 대상: {len(coins)}개 코인")

    # 기존 weekly_indicators 테이블 초기화
    conn.execute("DROP TABLE IF EXISTS weekly_indicators")
    conn.execute("""
        CREATE TABLE weekly_indicators (
            coin_id TEXT,
            date    TEXT,
            week    INTEGER,
            close   REAL,
            volume  REAL,
            mcap    REAL,
            ret     REAL,
            """ + ",\n            ".join(f"{c} REAL" for c in INDICATOR_COLS) + """,
            PRIMARY KEY (coin_id, date)
        )
    """)
    conn.commit()

    all_weekly = []
    for i, coin_id in enumerate(coins):
        df = pd.read_sql(
            "SELECT date, close, volume, mcap FROM daily_prices WHERE coin_id=? ORDER BY date",
            conn, params=(coin_id,)
        )
        if len(df) < 30:  # 최소 30일치 없으면 스킵
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
        cols = ["coin_id", "date", "week", "close", "volume", "mcap", "ret"] + INDICATOR_COLS
        result["date"] = result["date"].astype(str)
        result[cols].to_sql("weekly_indicators", conn, if_exists="append", index=False)
        conn.commit()
        log(f"주간 지표 저장 완료: {len(result):,}행 ({result['coin_id'].nunique()}개 코인)")
    else:
        log("저장할 데이터 없음")

    conn.close()

if __name__ == "__main__":
    main()
