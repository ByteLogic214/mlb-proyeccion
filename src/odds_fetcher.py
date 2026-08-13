"""
Odds fetcher and utilities. Supports reading odds from CSV and basic conversions.
The module intentionally avoids hard dependency on paid APIs; a placeholder for API integration
is included where you can drop the provider-specific implementation.

CSV format expected (simple): game_id, home_odds, away_odds (decimal) OR american odds columns.
"""

import csv
from typing import Dict

import pandas as pd
import numpy as np


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if np.isnan(american):
        return np.nan
    american = float(american)
    if american > 0:
        return 1.0 + american / 100.0
    elif american < 0:
        return 1.0 + 100.0 / (-american)
    else:
        return np.nan


def load_odds_csv(path: str, odds_type: str = 'decimal') -> pd.DataFrame:
    """Load odds CSV and standardize to decimal odds.

    Columns recognized:
      - game_id
      - home_odds, away_odds (decimal) OR
      - home_american, away_american (american)

    Returns DataFrame with game_id, home_odds(decimal), away_odds(decimal)
    """
    df = pd.read_csv(path)

    # detect american columns
    if 'home_american' in df.columns or 'away_american' in df.columns:
        df['home_odds'] = df['home_american'].apply(american_to_decimal)
        df['away_odds'] = df['away_american'].apply(american_to_decimal)
    else:
        # assume decimal
        if 'home_odds' not in df.columns and 'home_decimal' in df.columns:
            df['home_odds'] = df['home_decimal']
        if 'away_odds' not in df.columns and 'away_decimal' in df.columns:
            df['away_odds'] = df['away_decimal']

    # Ensure required columns
    if 'game_id' not in df.columns:
        raise ValueError('Odds CSV must contain game_id')

    return df[['game_id', 'home_odds', 'away_odds']]


# Placeholder for API integration

def fetch_odds_from_api(provider: str, credentials: Dict, date: str = None) -> pd.DataFrame:
    """Fetch odds from a provider. This is a placeholder where provider-specific code should go.

    Implementers should return a DataFrame with game_id, home_odds (decimal), away_odds (decimal).
    """
    raise NotImplementedError('API integration not implemented. Load from CSV using load_odds_csv.')
