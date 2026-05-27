"""
4_paper_trading.py — 페이퍼 트레이딩 추적
==========================================
매주 실행 시:
  1. 지난주 신호(latest_signal.json)의 롱 코인 → 실제 수익률 계산
  2. 결과를 paper_trading_log.csv에 누적 기록
  3. 누적 성과 요약 출력

흐름:
  토요일 실행 → 이번 주 신호 저장 (3_ctrend.py)
              → 지난주 신호 결과 확인 (4_paper_trading.py)
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH    = Path("data/prices.db")
SIGNAL_DIR = Path("signals")
LOG_PATH   = Path("signals/paper_trading_log.csv")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 지난주 신호 파일 찾기 ─────────────────────────────────
def find_last_signal():
    """signals/ 폴더에서 가장 최근 signal_*.csv 파일 2개 반환"""
    files = sorted(SIGNAL_DIR.glob("signal_*.csv"))
    if len(files) < 2:
        return None, None
    # 최신 = 이번 주, 그 전 = 지난주
    return files[-2], files[-1]

# ── 실제 수익률 계산 ──────────────────────────────────────
def calc_actual_returns(signal_csv, conn):
    """
    signal_csv: 지난주 신호 파일
    신호 날짜 이후 1주일간의 실제 수익률 계산
    """
    sig = pd.read_csv(signal_csv)
    sig_date = pd.to_datetime(signal_csv.stem.replace("signal_", ""))
    next_date = sig_date + timedelta(days=7)

    results = []
    for _, row in sig.iterrows():
        if row["signal"] not in ("long", "short"):
            continue

        coin_id = row["coin_id"]

        # 신호 날짜 직전 close
        price_before = pd.read_sql("""
            SELECT close FROM daily_prices
            WHERE coin_id=? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, conn, params=(coin_id, sig_date.strftime("%Y-%m-%d")))

        # 1주일 후 close
        price_after = pd.read_sql("""
            SELECT close, date FROM daily_prices
            WHERE coin_id=? AND date > ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, conn, params=(coin_id,
                           sig_date.strftime("%Y-%m-%d"),
                           next_date.strftime("%Y-%m-%d")))

        if price_before.empty or price_after.empty:
            continue

        p0  = price_before.iloc[0]["close"]
        p1  = price_after.iloc[0]["close"]
        ret = (p1 - p0) / p0

        # 롱은 수익률 그대로, 숏은 반대
        actual = ret if row["signal"] == "long" else -ret

        results.append({
            "signal_date": sig_date.strftime("%Y-%m-%d"),
            "eval_date":   price_after.iloc[0]["date"],
            "coin_id":     coin_id,
            "signal":      row["signal"],
            "price_entry": round(p0, 6),
            "price_exit":  round(p1, 6),
            "raw_ret":     round(ret,    4),
            "actual_ret":  round(actual, 4),
        })

    return pd.DataFrame(results)

# ── 주간 포트폴리오 수익률 계산 ───────────────────────────
def calc_weekly_port_ret(df):
    """롱/숏 동일 가중 평균"""
    if df.empty:
        return None
    return df["actual_ret"].mean()

# ── 누적 로그 업데이트 ────────────────────────────────────
def update_log(new_rows):
    if new_rows.empty:
        return

    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        # 중복 방지: 같은 signal_date + coin_id는 덮어쓰기
        existing = existing[
            ~existing["signal_date"].isin(new_rows["signal_date"].unique())
        ]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(LOG_PATH, index=False)
    log(f"로그 업데이트: {LOG_PATH} ({len(combined)}행)")

# ── 누적 성과 요약 ────────────────────────────────────────
def print_summary():
    if not LOG_PATH.exists():
        log("아직 기록 없음")
        return

    df = pd.read_csv(LOG_PATH)

    # 주간 포트폴리오 수익률
    weekly = df.groupby("signal_date")["actual_ret"].mean().reset_index()
    weekly.columns = ["signal_date", "port_ret"]
    weekly = weekly.sort_values("signal_date")

    r = weekly["port_ret"]
    if len(r) == 0:
        return

    cumret   = (1 + r).prod() - 1
    mean_w   = r.mean()
    std_w    = r.std()
    sharpe   = mean_w / std_w * np.sqrt(52) if std_w > 0 else 0
    win_rate = (r > 0).mean()
    best_w   = r.max()
    worst_w  = r.min()

    log("=" * 50)
    log("  📊 페이퍼 트레이딩 누적 성과")
    log("=" * 50)
    log(f"  추적 기간:      {weekly['signal_date'].iloc[0]} ~ {weekly['signal_date'].iloc[-1]}")
    log(f"  총 주수:        {len(r)}주")
    log(f"  평균 수익률:    {mean_w*100:+.2f}%/주")
    log(f"  누적 수익률:    {cumret*100:+.1f}%")
    log(f"  Sharpe (연환산): {sharpe:.2f}")
    log(f"  승률:           {win_rate*100:.1f}%")
    log(f"  최고 주:        {best_w*100:+.1f}%")
    log(f"  최저 주:        {worst_w*100:+.1f}%")
    log("=" * 50)

    # 최근 5주 상세
    log("\n  최근 5주 상세:")
    for _, row in weekly.tail(5).iterrows():
        bar = "▲" if row["port_ret"] > 0 else "▼"
        log(f"    {row['signal_date']}  {bar} {row['port_ret']*100:+.2f}%")

    # 이번 주 롱 코인별 수익률
    if len(df) > 0:
        latest_date = df["signal_date"].max()
        latest = df[df["signal_date"] == latest_date].sort_values(
            "actual_ret", ascending=False)
        log(f"\n  지난주 ({latest_date}) 코인별 결과:")
        for _, row in latest.iterrows():
            bar = "▲" if row["actual_ret"] > 0 else "▼"
            log(f"    {row['coin_id']:<15} [{row['signal']:>5}]  "
                f"{bar} {row['actual_ret']*100:+.2f}%")

# ── 메인 ─────────────────────────────────────────────────
def main():
    log("페이퍼 트레이딩 추적 시작")

    prev_signal, curr_signal = find_last_signal()

    if prev_signal is None:
        log("신호 파일이 2개 이상 필요합니다 (첫 주는 스킵)")
        print_summary()
        return

    log(f"평가 대상: {prev_signal.name}")

    conn = sqlite3.connect(DB_PATH)
    new_rows = calc_actual_returns(prev_signal, conn)
    conn.close()

    if new_rows.empty:
        log("수익률 계산 실패 — 가격 데이터 부족")
        print_summary()
        return

    port_ret = calc_weekly_port_ret(new_rows)
    log(f"이번 주 포트폴리오 수익률: {port_ret*100:+.2f}%")

    update_log(new_rows)
    print_summary()

if __name__ == "__main__":
    main()
