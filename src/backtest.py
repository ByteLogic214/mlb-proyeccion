"""
Backtesting harness for MLB projections.
- Loads projections and results
- Computes calibration metrics (Brier, LogLoss)
- Performs recalibration (Platt/Isotonic)
- Simulates simple betting strategies using odds (expected value)

Usage:
    python src/backtest.py --projections data/projections.csv --results data/historical_results.csv --odds data/odds.csv

If you don't have historical files, the module provides functions you can call programmatically.
"""

import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def load_joined_data(projections_path: str, results_path: Optional[str] = None) -> pd.DataFrame:
    """Load projection file and optionally join with actual results.

    Expected projections CSV columns: game_id, prob_home_win (or prob_away_win)
    Expected results CSV columns: game_id, home_score, away_score

    Returns a DataFrame with at least: game_id, pred_prob, actual (1 if home won else 0)
    """
    proj = pd.read_csv(projections_path)

    # normalize prediction column
    if 'prob_home_win' in proj.columns:
        proj['pred_prob'] = proj['prob_home_win']
    elif 'prob_home' in proj.columns:
        proj['pred_prob'] = proj['prob_home']
    elif 'pred_prob' in proj.columns:
        proj['pred_prob'] = proj['pred_prob']
    else:
        raise ValueError('Projections CSV must contain a prob_home_win or pred_prob column')

    df = proj.copy()

    if results_path:
        res = pd.read_csv(results_path)
        if 'game_id' not in res.columns:
            raise ValueError('Results CSV must contain game_id')
        # determine actual outcome
        if 'home_score' in res.columns and 'away_score' in res.columns:
            res['actual'] = (res['home_score'] > res['away_score']).astype(int)
        elif 'winner' in res.columns:
            # winner expected as 'home'/'away' or team ids
            res['actual'] = res['winner'].apply(lambda x: 1 if str(x).lower() in ('home', 'h') else 0)
        else:
            raise ValueError('Results CSV must contain home_score/away_score or winner column')

        df = df.merge(res[['game_id', 'actual']], on='game_id', how='left')

    return df


def evaluate_calibration(df: pd.DataFrame, prob_col: str = 'pred_prob', actual_col: str = 'actual') -> dict:
    """Return calibration metrics: Brier and LogLoss and simple reliability table."""
    mask = df[prob_col].notna() & df[actual_col].notna()
    sub = df.loc[mask]
    if sub.empty:
        return {}

    brier = float(brier_score_loss(sub[actual_col], sub[prob_col]))
    # logloss requires labels in {0,1} and probabilities in (0,1)
    eps = 1e-15
    probs = np.clip(sub[prob_col].astype(float), eps, 1 - eps)
    ll = float(log_loss(sub[actual_col], probs))

    # reliability: bucket by deciles
    sub['_bucket'] = pd.qcut(sub[prob_col], q=10, duplicates='drop')
    reliability = sub.groupby('_bucket').apply(lambda g: pd.Series({'mean_pred': g[prob_col].mean(), 'empirical': g[actual_col].mean(), 'count': len(g)}))
    reliability = reliability.reset_index()

    return {'brier': brier, 'logloss': ll, 'reliability': reliability}


def platt_recalibrate(train_df: pd.DataFrame, prob_col: str = 'pred_prob', actual_col: str = 'actual'):
    """Fit a logistic regression (Platt scaling) to map predicted probs to calibrated probs."""
    mask = train_df[prob_col].notna() & train_df[actual_col].notna()
    X = train_df.loc[mask, prob_col].values.reshape(-1, 1)
    y = train_df.loc[mask, actual_col].values

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X, y)

    def calibrate(probs):
        probs = np.array(probs).reshape(-1, 1)
        preds = lr.predict_proba(probs)[:, 1]
        return preds

    return calibrate, lr


def isotonic_recalibrate(train_df: pd.DataFrame, prob_col: str = 'pred_prob', actual_col: str = 'actual'):
    mask = train_df[prob_col].notna() & train_df[actual_col].notna()
    X = train_df.loc[mask, prob_col]
    y = train_df.loc[mask, actual_col]

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(X, y)

    def calibrate(probs):
        return ir.predict(probs)

    return calibrate, ir


def simulate_betting(df: pd.DataFrame, prob_col: str, odds_col: str, side: str = 'back', stake: float = 1.0, min_ev: float = 0.01) -> dict:
    """Simulate betting on rows where expected value >= min_ev.

    odds_col expected to be decimal odds (e.g. 2.5). side ignored for now (assume back the outcome target of prob_col).
    Returns dict with profit, ROI, bets_placed etc.
    """
    d = df.copy()
    eps = 1e-12
    d[prob_col] = d[prob_col].astype(float)
    d[odds_col] = d[odds_col].astype(float)

    # implied probability from decimal odds
    d['implied'] = 1.0 / np.clip(d[odds_col].values, eps, None)

    # expected return per unit stake: prob * (odds - 1) - (1 - prob) * 1 = prob*odds -1
    d['ev_per_unit'] = d[prob_col] * d[odds_col] - 1.0

    # pick bets where ev_per_unit >= min_ev
    bets = d[d['ev_per_unit'] >= min_ev].copy()
    if bets.empty:
        return {'total_profit': 0.0, 'roi': 0.0, 'n_bets': 0, 'details': bets}

    # simulate results: need actual column (1 if target event occurred)
    if 'actual' not in bets.columns:
        raise ValueError('Simulation requires actual column in DataFrame')

    bets['stake'] = stake
    # profit per bet
    bets['profit'] = bets.apply(lambda r: (r[odds_col] - 1) * r['stake'] if r[prob_col] == r['actual'] and r['actual'] == 1 else ((-1) * r['stake'] if r['actual'] == 0 else ((r[odds_col] - 1) * r['stake'] if r['actual']==1 else -r['stake'])), axis=1)
    # The lambda above is generic but we assume target event is actual==1 meaning the predicted event happened
    # Simpler: if actual==1 profit = (odds-1)*stake else -stake. We'll use that.
    bets['profit'] = bets.apply(lambda r: (r[odds_col] - 1) * r['stake'] if r['actual'] == 1 else -r['stake'], axis=1)

    total_profit = float(bets['profit'].sum())
    total_staked = float((bets['stake']).sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0

    return {'total_profit': total_profit, 'roi': roi, 'n_bets': len(bets), 'details': bets}


def main(args):
    df = load_joined_data(args.projections, args.results)

    metrics = evaluate_calibration(df)
    print('Brier:', metrics.get('brier'))
    print('LogLoss:', metrics.get('logloss'))

    # Example: fit isotonic on the full set and apply
    if args.do_recalibrate:
        calibrate_fn, model = isotonic_recalibrate(df)
        df['calibrated_prob'] = calibrate_fn(df['pred_prob'].fillna(0.5))
        metrics_cal = evaluate_calibration(df.rename(columns={'calibrated_prob': 'pred_prob'}), prob_col='pred_prob', actual_col='actual')
        print('After isotonic - Brier:', metrics_cal.get('brier'))

    # If odds provided, load and run a simple EV simulation
    if args.odds:
        odds = pd.read_csv(args.odds)
        # expect odds.csv to have game_id and decimal odds for home in column home_odds
        merged = df.merge(odds[['game_id', 'home_odds']], on='game_id', how='left')
        results = simulate_betting(merged, prob_col='pred_prob', odds_col='home_odds', stake=1.0, min_ev=args.min_ev)
        print('Betting simulation:', results['n_bets'], 'bets => profit:', results['total_profit'], 'ROI:', results['roi'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--projections', required=True)
    parser.add_argument('--results', required=False)
    parser.add_argument('--odds', required=False)
    parser.add_argument('--min_ev', type=float, default=0.01)
    parser.add_argument('--do_recalibrate', action='store_true')
    args = parser.parse_args()
    main(args)
