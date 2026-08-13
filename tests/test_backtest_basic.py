import pandas as pd
import numpy as np
from src.backtest import evaluate_calibration, simulate_betting


def test_evaluate_and_simulate():
    # small synthetic dataset
    df = pd.DataFrame({
        'game_id': [1,2,3,4],
        'pred_prob': [0.70, 0.60, 0.40, 0.30],
        'actual': [1, 0, 0, 1],
        'home_odds': [1.6, 1.8, 2.5, 3.5]
    })

    metrics = evaluate_calibration(df)
    assert 'brier' in metrics
    assert metrics['brier'] >= 0.0

    sim = simulate_betting(df, prob_col='pred_prob', odds_col='home_odds', stake=1.0, min_ev=-1.0)
    # with min_ev negative we will bet on all, expect numeric results
    assert 'total_profit' in sim
    assert 'roi' in sim
