"""
3_ctrend.py — CTREND 신호 생성 및 검증
=========================================
논문 방식 그대로:
  1. 매주 각 코인의 28개 지표 → Elastic Net으로 수익률 예측
  2. 예측값 기준 상위 20% 코인 = 롱, 하위 20% = 숏
  3. 가치가중(시가총액) 포트폴리오 수익률 계산
  4. 결과를 signals/ 폴더에 저장
  5. CTREND.xlsx 저자 결과와 비교 (겹치는 구간)

[논문 vs 우리 구현 차이]
  논문: 3,000개 코인 / 2015~2022 / OHLCV 전체
  우리: 300개 코인 / 2022~현재 / Close+Volume only
  → out-of-sample 검증이 목적이므로 방향성 비교가 핵심
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

DB_PATH      = Path("data/prices.db")
SIGNALS_DIR  = Path("signals")
PAPER_CSV    = Path("data/ctrend_paper.csv")

INDICATOR_COLS = [
    "rsi", "stoch_rsi", "stoch_k", "stoch_d", "cci",
    "sma_3d", "sma_5d", "sma_10d", "sma_20d", "sma_50d", "sma_100d", "sma_200d",
    "ppo", "ppo_diff",
    "vsma_3d", "vsma_5d", "vsma_10d", "vsma_20d", "vsma_50d", "vsma_100d", "vsma_200d",
    "pvo", "pvo_diff",
    "chaikin",
    "boll_low", "boll_mid", "boll_high", "boll_width"
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ════════════════════════════════════════════════════
# Elastic Net (walk-forward)
# ════════════════════════════════════════════════════

def walkforward_enet(panel, min_train_weeks=52):
    """
    매 주차마다:
      - 과거 전체 데이터로 Elastic Net 훈련
      - 해당 주 지표값으로 수익률 예측
    반환: 주차별 코인별 예측값 DataFrame
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    weeks    = sorted(panel["week"].unique())
    results  = []

    for i, w in enumerate(weeks):
        if i < min_train_weeks:
            continue

        train = panel[panel["week"] < w].dropna(subset=INDICATOR_COLS + ["ret"])
        test  = panel[panel["week"] == w].dropna(subset=INDICATOR_COLS)

        if len(train) < 100 or len(test) < 5:
            continue

        X_train = train[INDICATOR_COLS].values
        y_train = train["ret"].values
        X_test  = test[INDICATOR_COLS].values

        # 표준화
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Winsorize y
        lo, hi  = np.percentile(y_train, [0.5, 99.5])
        y_train = np.clip(y_train, lo, hi)

        try:
            model   = ElasticNetCV(cv=5, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            preds   = model.predict(X_test)
        except Exception as e:
            log(f"  ElasticNet 오류 (week {w}): {e}")
            continue

        df_pred = test[["coin_id", "week", "date", "ret", "mcap"]].copy()
        df_pred["pred"] = preds
        results.append(df_pred)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ════════════════════════════════════════════════════
# 포트폴리오 수익률 계산
# ════════════════════════════════════════════════════

def calc_portfolio_returns(pred_df, quantile=0.2):
    """
    상위 quantile = 롱, 하위 quantile = 숏
    시가총액 가중 수익률
    """
    results = []

    for week, grp in pred_df.groupby("week"):
        n = len(grp)
        if n < 10:
            continue

        # 분위 경계
        lo_cut = grp["pred"].quantile(quantile)
        hi_cut = grp["pred"].quantile(1 - quantile)

        long_leg  = grp[grp["pred"] >= hi_cut].copy()
        short_leg = grp[grp["pred"] <= lo_cut].copy()

        if len(long_leg) == 0 or len(short_leg) == 0:
            continue

        # 시가총액 가중
        def vw_ret(df):
            w = df["mcap"] / df["mcap"].sum()
            return (w * df["ret"]).sum()

        long_ret  =  vw_ret(long_leg)
        short_ret =  vw_ret(short_leg)
        ls_ret    = long_ret - short_ret

        results.append({
            "week":       week,
            "date":       grp["date"].iloc[0],
            "long_ret":   long_ret,
            "short_ret":  short_ret,
            "ctrend_ret": ls_ret,
            "n_coins":    n,
            "n_long":     len(long_leg),
            "n_short":    len(short_leg),
        })

    return pd.DataFrame(results)

# ════════════════════════════════════════════════════
# 성과 요약
# ════════════════════════════════════════════════════

def summarize(ret_df):
    r = ret_df["ctrend_ret"].dropna()
    weeks = len(r)
    if weeks == 0:
        return {}

    ann_factor = 52  # 주간 → 연간
    mean_w  = r.mean()
    std_w   = r.std()
    sharpe  = (mean_w / std_w * np.sqrt(ann_factor)) if std_w > 0 else 0
    win_rate = (r > 0).mean()
    cumret  = (1 + r).prod() - 1

    return {
        "weeks":          weeks,
        "mean_weekly":    round(mean_w,  4),
        "std_weekly":     round(std_w,   4),
        "sharpe_annual":  round(sharpe,  3),
        "win_rate":       round(win_rate,3),
        "cumulative_ret": round(cumret,  4),
    }

# ════════════════════════════════════════════════════
# 저자 결과와 비교
# ════════════════════════════════════════════════════

def compare_with_paper(our_ret_df):
    if not PAPER_CSV.exists():
        log("ctrend_paper.csv 없음 → 비교 스킵")
        return

    paper = pd.read_csv(PAPER_CSV)
    paper["date"] = pd.to_datetime(paper["date"])

    ours = our_ret_df.copy()
    ours["date"] = pd.to_datetime(ours["date"])

    merged = pd.merge(paper, ours[["date","ctrend_ret"]],
                      on="date", how="inner")

    if len(merged) < 4:
        log(f"겹치는 구간 {len(merged)}주 → 비교 불가")
        return

    corr = merged["CTREND"].corr(merged["ctrend_ret"])
    log(f"논문 vs 우리 상관계수 (겹치는 {len(merged)}주): {corr:.3f}")
    return corr

# ════════════════════════════════════════════════════
# 이번 주 신호 저장
# ════════════════════════════════════════════════════

def save_latest_signal(pred_df):
    """가장 최근 주차의 롱/숏 신호를 signals/ 폴더에 저장"""
    SIGNALS_DIR.mkdir(exist_ok=True)

    latest_week = pred_df["week"].max()
    latest = pred_df[pred_df["week"] == latest_week].copy()
    latest = latest.sort_values("pred", ascending=False)

    n = len(latest)
    cut = max(1, int(n * 0.2))

    latest["signal"] = "neutral"
    latest.iloc[:cut,  latest.columns.get_loc("signal")] = "long"
    latest.iloc[-cut:, latest.columns.get_loc("signal")] = "short"

    # CSV 저장
    date_str = latest["date"].iloc[0][:10] if len(latest) > 0 else "unknown"
    out_path = SIGNALS_DIR / f"signal_{date_str}.csv"
    latest[["coin_id","pred","signal","mcap","ret"]].to_csv(out_path, index=False)
    log(f"신호 저장: {out_path} ({cut}롱 / {cut}숏)")

    # latest_signal.json (가장 최신 신호)
    long_coins  = latest[latest["signal"]=="long"]["coin_id"].tolist()
    short_coins = latest[latest["signal"]=="short"]["coin_id"].tolist()
    with open(SIGNALS_DIR / "latest_signal.json", "w") as f:
        json.dump({
            "date":        date_str,
            "week":        int(latest_week),
            "long_coins":  long_coins,
            "short_coins": short_coins,
            "n_total":     n,
        }, f, indent=2, ensure_ascii=False)

    log(f"롱 신호 상위 5: {long_coins[:5]}")

# ════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════

def main():
    conn  = sqlite3.connect(DB_PATH)
    panel = pd.read_sql("""
        SELECT coin_id, date, week, close, volume, mcap, ret,
               """ + ", ".join(INDICATOR_COLS) + """
        FROM weekly_indicators
        ORDER BY week, coin_id
    """, conn)
    conn.close()

    log(f"패널 데이터: {len(panel):,}행 / {panel['coin_id'].nunique()}개 코인 / {panel['week'].nunique()}주")

    if panel.empty:
        log("데이터 없음 — 1_collect.py와 2_indicators.py를 먼저 실행하세요")
        return

    # Section III.A 필터: 시가총액 $1M 미만 제거
    panel = panel[panel["mcap"] > 1_000_000]
    log(f"시가총액 필터 후: {panel['coin_id'].nunique()}개 코인")

    # Walk-forward Elastic Net
    log("Walk-forward Elastic Net 실행 중...")
    pred_df = walkforward_enet(panel, min_train_weeks=52)

    if pred_df.empty:
        log("예측 결과 없음 — 데이터가 52주(1년) 이상 필요합니다")
        log(f"현재 보유 주수: {panel['week'].nunique()}주")
        return

    # 포트폴리오 수익률
    ret_df = calc_portfolio_returns(pred_df)
    log(f"포트폴리오 계산 완료: {len(ret_df)}주")

    # 성과 요약
    summary = summarize(ret_df)
    log("\n── 성과 요약 ──────────────────────────")
    for k, v in summary.items():
        log(f"  {k:20s}: {v}")
    log("──────────────────────────────────────")

    # 논문 평균 +3.87%/주와 비교
    if "mean_weekly" in summary:
        paper_mean = 0.0387
        ours_mean  = summary["mean_weekly"]
        diff       = ours_mean - paper_mean
        log(f"\n  논문 평균 수익률: +{paper_mean*100:.2f}%/주")
        log(f"  우리 평균 수익률: {ours_mean*100:+.2f}%/주")
        log(f"  차이: {diff*100:+.2f}%p")

    # 저자 결과와 상관계수 비교
    compare_with_paper(ret_df)

    # 최신 신호 저장
    save_latest_signal(pred_df)

    # 전체 수익률 이력 저장
    SIGNALS_DIR.mkdir(exist_ok=True)
    ret_df.to_csv(SIGNALS_DIR / "ctrend_returns.csv", index=False)
    log(f"\n수익률 이력 저장: signals/ctrend_returns.csv")

    # 성과 요약 JSON
    summary["last_updated"] = datetime.now().isoformat()
    summary["paper_mean_weekly"] = 0.0387
    with open(SIGNALS_DIR / "performance.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("성과 요약 저장: signals/performance.json")

if __name__ == "__main__":
    main()
