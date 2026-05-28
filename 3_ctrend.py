"""
3_ctrend.py — CTREND 신호 생성 및 검증
=========================================
[v3 수정]
  - NaN 30% 이하 컬럼만 사용 (참여 코인 수 증가)
  - 사용 가능한 지표가 주차마다 동적으로 결정됨
  - 최소 10개 지표 없으면 해당 주차 스킵
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

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
    매 주차마다:
      - NaN 비율 30% 이하인 지표만 사용 (동적 선택)
      - X(t주 지표) → y(t+1주 수익률)
      - 최소 10개 지표 없으면 스킵
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    panel = panel.copy()
    panel["ret_next"] = panel.groupby("coin_id")["ret"].shift(-1)

    weeks   = sorted(panel["week"].unique())
    results = []
    skipped = 0

    for i, w in enumerate(weeks):
        if i < min_train_weeks:
            continue

        train_all = panel[panel["week"] < w].copy()
        test_all  = panel[panel["week"] == w].copy()

        # NaN 30% 이하인 지표만 선택
        use_cols = [c for c in INDICATOR_COLS
                    if train_all[c].isna().mean() < 0.3]

        if len(use_cols) < 10:
            skipped += 1
            continue

        train = train_all.dropna(subset=use_cols + ["ret_next"])
        test  = test_all.dropna(subset=use_cols)

        if len(train) < 50 or len(test) < 5:
            skipped += 1
            continue

        X_train = train[use_cols].values
        y_train = train["ret_next"].values
        X_test  = test[use_cols].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Winsorize y (상하 0.5%)
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
        df_pred["pred"]     = preds
        df_pred["n_cols"]   = len(use_cols)
        df_pred["n_train"]  = len(train)
        results.append(df_pred)

    if skipped > 0:
        log(f"  스킵된 주차: {skipped}개 (지표/데이터 부족)")

    if not results:
        return pd.DataFrame(), panel
    return pd.concat(results, ignore_index=True), panel

# ── 포트폴리오 수익률 계산 ────────────────────────────────
def calc_portfolio_returns(pred_df, panel, quantile=0.2):
    # 다음 주 실제 수익률 매핑
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
            "n_cols":     int(grp["n_cols"].iloc[0]),
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

    # 포트폴리오 참여 코인 수 평균
    avg_n = ret_df["n_coins"].mean() if "n_coins" in ret_df.columns else 0
    avg_cols = ret_df["n_cols"].mean() if "n_cols" in ret_df.columns else 0

    return {
        "weeks":           len(r),
        "mean_weekly":     round(mean_w,   4),
        "std_weekly":      round(std_w,    4),
        "sharpe_annual":   round(sharpe,   3),
        "win_rate":        round(win_rate, 3),
        "cumulative_ret":  round(cumret,   4),
        "avg_coins_pw":    round(avg_n,    1),
        "avg_indicators":  round(avg_cols, 1),
    }

# ── 저자 결과와 비교 ──────────────────────────────────────
def compare_with_paper(our_ret_df):
    if not PAPER_CSV.exists():
        log("ctrend_paper.csv 없음 → 비교 스킵")
        return

    paper = pd.read_csv(PAPER_CSV, index_col=None, sep=None, engine="python")
    paper.columns = paper.columns.str.strip()
    if "date" not in paper.columns:
        first_col = paper.columns[0]
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

    n   = len(latest)
    cut = max(1, int(n * 0.2))
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
            "date":        date_str,
            "week":        int(latest_week),
            "long_coins":  long_coins,
            "short_coins": short_coins,
            "n_total":     n,
            "n_long":      cut,
            "n_short":     cut,
        }, f, indent=2, ensure_ascii=False)

    log(f"신호 저장 완료: {date_str} | 롱 {cut}개 / 숏 {cut}개 (전체 {n}개 중)")
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

    # 시가총액 하위 10% 제거
    mcap_threshold = panel["mcap"].quantile(0.10)
    panel = panel[panel["mcap"] > mcap_threshold]
    log(f"시가총액 필터 후: {panel['coin_id'].nunique()}개 코인 (하위 10% 제거)")

    log("Walk-forward Elastic Net 실행 중...")
    pred_df, panel_with_next = walkforward_enet(panel, min_train_weeks=52)

    if pred_df.empty:
        log(f"예측 결과 없음 — 현재 {panel['week'].nunique()}주 보유")
        return

    # 포트폴리오 참여 코인 수 분포 확인
    n_dist = pred_df.groupby("week").size()
    log(f"주차별 참여 코인 수: 평균={n_dist.mean():.1f}, "
        f"최소={n_dist.min()}, 최대={n_dist.max()}")

    ret_df = calc_portfolio_returns(pred_df, panel_with_next)
    log(f"포트폴리오 계산 완료: {len(ret_df)}주")

    summary = summarize(ret_df)
    log("\n── 성과 요약 ──────────────────────────")
    for k, v in summary.items():
        log(f"  {k:22s}: {v}")
    log("──────────────────────────────────────")

    if "mean_weekly" in summary:
        log(f"\n  논문 평균 수익률:  +3.87%/주")
        log(f"  우리 평균 수익률:  {summary['mean_weekly']*100:+.2f}%/주")

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
