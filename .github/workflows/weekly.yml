name: CTREND Weekly Pipeline

on:
  schedule:
    - cron: '0 0 * * 6'
  workflow_dispatch:

jobs:
  pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install requests pandas numpy scikit-learn openpyxl

      - name: Step 1 — 데이터 수집 (Binance + CoinGecko 시가총액)
        env:
          COINGECKO_API_KEY: ${{ secrets.COINGECKO_API_KEY }}
        run: python 1_collect.py

      - name: Step 2 — 지표 계산
        run: python 2_indicators.py

      - name: Step 3 — CTREND 신호 생성
        run: python 3_ctrend.py

      - name: Step 4 — 페이퍼 트레이딩 추적
        run: python 4_paper_trading.py

      - name: Commit results
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/prices.db signals/ || true
          git diff --staged --quiet || \
            git commit -m "chore: weekly update $(date +'%Y-%m-%d')"
          git push
