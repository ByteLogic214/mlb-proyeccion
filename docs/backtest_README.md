# Backtest README

This folder includes scripts to backtest and evaluate projections and a simple odds/EV simulation.

Usage examples:

1) Basic calibration evaluation (when you have projections and actuals):

    python src/backtest.py --projections data/projections.csv --results data/historical_results.csv

2) Evaluate and simulate betting when you have a CSV of market odds:

    python src/backtest.py --projections data/projections.csv --results data/historical_results.csv --odds data/odds.csv --min_ev 0.02

Files added:
- src/backtest.py: main harness for calibration & betting simulation
- src/odds_fetcher.py: helpers to read/normalize odds data

Notes:
- Odds CSV expected columns: game_id, home_odds (decimal). If using american odds, use columns home_american/away_american.
- The betting simulator uses a simple expected value filter (EV >= min_ev) and stakes a fixed unit per bet.
- Before running large-scale simulations, run the scripts on a small sample to verify formats.
