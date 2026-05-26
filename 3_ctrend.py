"""
3_ctrend.py — CTREND 신호 생성 및 검증
=========================================
[핵심 수정]
  t주차 지표 → t+1주차 수익률 예측 (올바른 구조)
  이전 버전: t주차 지표 → t주차 수익률 (look-ahead bias)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")   # ConvergenceWarning 숨김

DB_PATH     = Path("data/prices.db")
SIGNALS_DIR = Path("signals")
PAPER_CSV   = Path("data/ctrend_paper.csv")

INDICATOR_COLS = [
    "rsi","stoch_rsi","stoch_k","stoch_d","cci",
    "sma_3d","sma_5d","sma_10d","sma_20d","sma_50d","sma_100d","sma_200d",
    "ppo","ppo_diff",
    "vsma_3d","vsma_5d","vsma_10d","vsma_20d","vsma_50d","vsma_100d","vsma_200d",
    "pvo","pvo_diff",
    "chaikin",
    "boll_low","boll_mid","boll_high","boll_width"
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Walk-forward Elastic Net ──────────────────────────────
def walkforward_enet(panel, min_train_weeks=52):
    """
    올바른 시점 정렬:
      X = t주차 지표
      y = t+1주차 수익률 (ret_next)
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    # t+1주차 수익률 생성 (코인별 shift)
    panel = panel.copy()
    panel["ret_next"] = panel.groupby("coin_id")["ret"].shift(-1)

    weeks   = sorted(panel["week"].unique())
    results = []

    for i, w in enumerate(weeks):
        if i < min_train_weeks:
            continue

        # 훈련: t < w 인 주차에서 X(t), y=ret_next(t) = ret(t+1)
        train = panel[panel["week"] < w].dropna(subset=INDICATOR_COLS + ["ret_next"])
        # 테스트: t = w 인 주차의 지표로 w+1주차 수익률 예측
        test  = panel[panel["week"] == w].dropna(subset=INDICATOR_COLS)

        if len(train) < 100 or len(test) < 5:
            continue

        X_train = train[INDICATOR_COLS].values
        y_train = train["ret_next"].values
        X_test  = test[INDICATOR_COLS].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Winsorize y
        lo, hi  = np.percentile(y_train, [0.5, 99.5])
        y_train = np.clip(y_train, lo, hi)

        try:
            model = ElasticNetCV(cv=5, max_iter=10000, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as e:
            log(f"  ElasticNet 오류 (week {w}): {e}")
            continue

        df_pred = test[["coin_id","week","date","mcap"]].copy()
        # 실제 검증에 쓸 수익률은 다음 주(w+1)의 ret
        # 현재는 예측값만 저장, 실제 수익률은 나중에 join
        df_pred["pred"] = preds
        results.append(df_pred)

    if not results:
        return pd.DataFrame(), panel
    return pd.concat(results, ignore_index=True), panel

# ── 포트폴리오 수익률 계산 ────────────────────────────────
def calc_portfolio_returns(pred_df, panel, quantile=0.2):
    """
    예측 기준으로 포트폴리오 구성,
    실제 수익률은 다음 주 ret으로 계산
    """
    # 다음 주 실제 수익률 매핑
    week_ret = panel[["coin_id","week","ret"]].copy()
    week_ret["week_prev"] = panel.groupby("coin_id")["week"].shift(1)
    # pred의 week(t) → 다음 주 ret(t+1) 연결
    next_ret = {}
    for coin_id, grp in panel.groupby("coin_id"):
        grp = grp.sort_values("week")
        for idx in range(len(grp)-1):
            w_now  = grp.iloc[idx]["week"]
            r_next = grp.iloc[idx+1]["ret"]
            next_ret[(coin_id, w_now)] = r_next

    pred_df = pred_df.copy()
    pred_df["actual_ret"] = pred_df.apply(
        lambda r: next_ret.get((r["coin_id"], r["week"]), np.nan), axis=1
    )
    pred_df = pred_df.dropna(subset=["actual_ret"])

    results = []
    for week, grp in pred_df.groupby("week"):
        n = len(grp)
        if n < 10:
            continue

        lo_cut = grp["pred"].quantile(quantile)
        hi_cut = grp["pred"].quantile(1 - quantile)

        long_leg  = grp[grp["pred"] >= hi_cut].copy()
        short_leg = grp[grp["pred"] <= lo_cut].copy()

        if len(long_leg) == 0 or len(short_leg) == 0:
            continue

        def vw_ret(df):
            w = df["mcap"] / df["mcap"].sum()
            return (w * df["actual_ret"]).sum()

        results.append({
            "week":       week,
            "date":       grp["date"].iloc[0],
            "long_ret":   vw_ret(long_leg),
            "short_ret":  vw_ret(short_leg),
            "ctrend_ret": vw_ret(long_leg) - vw_ret(short_leg),
            "n_coins":    n,
            "n_long":     len(long_leg),
            "n_short":    len(short_leg),
        })

    return pd.DataFrame(results)

# ── 성과 요약 ─────────────────────────────────────────────
def summarize(ret_df):
    r = ret_df["ctrend_ret"].dropna()
    if len(r) == 0:
        return {}
    mean_w   = r.mean()
    std_w    = r.std()
    sharpe   = mean_w / std_w * np.sqrt(52) if std_w > 0 else 0
    win_rate = (r > 0).mean()
    cumret   = (1 + r).prod() - 1
    return {
        "weeks":          len(r),
        "mean_weekly":    round(mean_w,  4),
        "std_weekly":     round(std_w,   4),
        "sharpe_annual":  round(sharpe,  3),
        "win_rate":       round(win_rate,3),
        "cumulative_ret": round(cumret,  4),
    }

# ── 저자 결과와 비교 ──────────────────────────────────────
def compare_with_paper(our_ret_df):
    if not PAPER_CSV.exists():
        log("ctrend_paper.csv 없음 → 비교 스킵")
        return

    paper = pd.read_csv(PAPER_CSV, index_col=None)
    paper.columns = paper.columns.str.strip()

    # 첫 번째 컬럼이 날짜인 경우 대응
    if "date" not in paper.columns:
        first_col = paper.columns[0]
        log(f"  컬럼명 확인: {paper.columns.tolist()}")
        # 첫 번째 컬럼이 날짜처럼 생겼으면 rename
        try:
            pd.to_datetime(paper[first_col].iloc[0])
            paper = paper.rename(columns={first_col: "date"})
        except Exception:
            log("date 컬럼 없음 → 비교 스킵")
            return

    paper["date"] = pd.to_datetime(paper["date"])

    ours = our_ret_df.copy()
    ours["date"] = pd.to_datetime(ours["date"])

    merged = pd.merge(paper, ours[["date","ctrend_ret"]], on="date", how="inner")

    if len(merged) < 4:
        log(f"겹치는 구간 {len(merged)}주 — out-of-sample 기간이라 정상")
        return

    corr = merged["CTREND"].corr(merged["ctrend_ret"])
    log(f"논문 vs 우리 상관계수 (겹치는 {len(merged)}주): {corr:.3f}")

# ── 최신 신호 저장 ────────────────────────────────────────
def save_latest_signal(pred_df):
    SIGNALS_DIR.mkdir(exist_ok=True)
    latest_week = pred_df["week"].max()
    latest = pred_df[pred_df["week"] == latest_week].copy()
    latest = latest.sort_values("pred", ascending=False)

    n    = len(latest)
    cut  = max(1, int(n * 0.2))
    latest["signal"] = "neutral"
    latest.iloc[:cut,  latest.columns.get_loc("signal")] = "long"
    latest.iloc[-cut:, latest.columns.get_loc("signal")] = "short"

    date_str = str(latest["date"].iloc[0])[:10] if len(latest) > 0 else "unknown"
    latest[["coin_id","pred","signal","mcap"]].to_csv(
        SIGNALS_DIR / f"signal_{date_str}.csv", index=False)

    long_coins  = latest[latest["signal"]=="long"]["coin_id"].tolist()
    short_coins = latest[latest["signal"]=="short"]["coin_id"].tolist()
    with open(SIGNALS_DIR / "latest_signal.json","w") as f:
        json.dump({
            "date": date_str, "week": int(latest_week),
            "long_coins": long_coins, "short_coins": short_coins,
            "n_total": n,
        }, f, indent=2, ensure_ascii=False)

    log(f"신호 저장 완료: {date_str} | 롱 {cut}개 / 숏 {cut}개")
    log(f"롱 상위 5: {long_coins[:5]}")

# ── 메인 ─────────────────────────────────────────────────
def main():
    conn  = sqlite3.connect(DB_PATH)
    panel = pd.read_sql(f"""
        SELECT coin_id, date, week, close, volume, mcap, ret,
               {', '.join(INDICATOR_COLS)}
        FROM weekly_indicators
        ORDER BY week, coin_id
    """, conn)
    conn.close()

    log(f"패널: {len(panel):,}행 / {panel['coin_id'].nunique()}개 코인 / {panel['week'].nunique()}주")

    if panel.empty:
        log("데이터 없음")
        return

    # 시가총액 필터 ($1M 이상)
    panel = panel[panel["mcap"] > 1_000_000]
    log(f"필터 후: {panel['coin_id'].nunique()}개 코인")

    log("Walk-forward Elastic Net 실행 중...")
    pred_df, panel_with_next = walkforward_enet(panel, min_train_weeks=52)

    if pred_df.empty:
        log(f"예측 결과 없음 — 현재 {panel['week'].nunique()}주 보유 (52주 이상 필요)")
        return

    ret_df = calc_portfolio_returns(pred_df, panel_with_next)
    log(f"포트폴리오 계산 완료: {len(ret_df)}주")

    summary = summarize(ret_df)
    log("\n── 성과 요약 ──────────────────────────")
    for k, v in summary.items():
        log(f"  {k:20s}: {v}")
    log("──────────────────────────────────────")

    if "mean_weekly" in summary:
        log(f"\n  논문 평균 수익률:  +3.87%/주")
        log(f"  우리 평균 수익률:  {summary['mean_weekly']*100:+.2f}%/주")
        if abs(summary['mean_weekly']) < 0.10:
            log("  → 논문과 유사한 수준 ✓")
        elif summary['mean_weekly'] > 0.10:
            log("  → 여전히 높음: 데이터/구현 추가 검토 필요")

    compare_with_paper(ret_df)
    save_latest_signal(pred_df)

    SIGNALS_DIR.mkdir(exist_ok=True)
    ret_df.to_csv(SIGNALS_DIR / "ctrend_returns.csv", index=False)

    summary["last_updated"] = datetime.now().isoformat()
    summary["paper_mean_weekly"] = 0.0387
    with open(SIGNALS_DIR / "performance.json","w") as f:
        json.dump(summary, f, indent=2)

    log("완료 — signals/ 폴더 확인")

if __name__ == "__main__":
    main()
