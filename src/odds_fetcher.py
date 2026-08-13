"""
Odds fetcher and utilities. Supports reading odds from CSV and basic conversions.
The module intentionally avoids hard dependency on paid APIs; it includes an implementation for The Odds API (https://the-odds-api.com/) using an API key from environment variable ODDS_API_KEY.

CSV format expected (simple): game_id, home_odds, away_odds (decimal) OR american odds columns.
"""

import os
import time
import requests
from typing import Dict

import pandas as pd
import numpy as np


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    try:
        if np.isnan(american):
            return np.nan
    except Exception:
        pass
    try:
        american = float(american)
    except Exception:
        return np.nan
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
    if 'game_id' not in df.columns and not (('home_team' in df.columns) and ('away_team' in df.columns)):
        raise ValueError('Odds CSV must contain game_id or home_team and away_team')

    # If no game_id, try to create one by concatenating teams
    if 'game_id' not in df.columns:
        df['game_id'] = df['away_team'].astype(str) + '_vs_' + df['home_team'].astype(str)

    return df[['game_id', 'home_team', 'away_team', 'home_odds', 'away_odds']].fillna(np.nan)


def fetch_odds_from_api(provider: str = 'the-odds-api', credentials: Dict = None, date: str = None, regions: str = 'us', markets: str = 'h2h', odds_format: str = 'decimal') -> pd.DataFrame:
    """Fetch odds from a provider. Currently implemented: The Odds API (the-odds-api.com).

    Returns a DataFrame with columns: home_team, away_team, home_odds (decimal), away_odds (decimal), commence_time, bookie
    
    Requires environment variable ODDS_API_KEY to be set if credentials is None.
    """
    if provider != 'the-odds-api':
        raise NotImplementedError('Only the-odds-api provider is implemented')

    api_key = None
    if credentials and 'api_key' in credentials:
        api_key = credentials['api_key']
    else:
        api_key = os.environ.get('ODDS_API_KEY')

    if not api_key:
        raise ValueError('ODDS_API_KEY not found in environment or credentials')

    sport_key = 'baseball_mlb'
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'dateFormat': 'iso'
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f'Error fetching odds from The Odds API: {e}')

    rows = []
    for ev in events:
        # The Odds API v4 includes home_team, away_team, commence_time, and bookmakers
        home = ev.get('home_team')
        away = ev.get('away_team')
        commence = ev.get('commence_time')

        # prefer the first bookmaker or the one with best last_update
        bookmakers = ev.get('bookmakers', [])
        if not bookmakers:
            continue

        # pick the bookmaker with the most markets (likely complete)
        bookmakers_sorted = sorted(bookmakers, key=lambda b: len(b.get('markets', [])), reverse=True)
        bookmaker = bookmakers_sorted[0]
        bookie_key = bookmaker.get('key')

        # find h2h market
        market = None
        for m in bookmaker.get('markets', []):
            if m.get('key') == 'h2h' or m.get('key') == markets:
                market = m
                break

        if not market:
            # fallback: try first market
            market = bookmaker.get('markets', [None])[0]

        if not market:
            continue

        outcomes = market.get('outcomes', [])
        # outcomes usually have [{'name': 'Team A', 'price': 1.9}, ...]
        home_odds = np.nan
        away_odds = np.nan
        for o in outcomes:
            name = o.get('name')
            price = o.get('price')
            if name is None or price is None:
                continue
            if name.strip().lower() == home.strip().lower():
                home_odds = float(price)
            elif name.strip().lower() == away.strip().lower():
                away_odds = float(price)

        # If names don't match due to abbreviations, try mapping by contains
        if (pd.isna(home_odds) or pd.isna(away_odds)) and outcomes:
            for o in outcomes:
                name = o.get('name')
                price = o.get('price')
                if name is None or price is None:
                    continue
                lname = name.strip().lower()
                if home and lname in home.lower():
                    home_odds = float(price)
                if away and lname in away.lower():
                    away_odds = float(price)

        # If still NaN, try to pick the two outcomes by order (best effort)
        if pd.isna(home_odds) or pd.isna(away_odds):
            try:
                if len(outcomes) >= 2:
                    # assume first is home/away order unknown; try to assign by comparing team names
                    o1 = outcomes[0]
                    o2 = outcomes[1]
                    # assign by whichever matches partially
                    p1 = o1.get('price')
                    p2 = o2.get('price')
                    home_odds = float(p1)
                    away_odds = float(p2)
                else:
                    home_odds = np.nan
                    away_odds = np.nan
            except Exception:
                home_odds = np.nan
                away_odds = np.nan

        rows.append({
            'home_team': home,
            'away_team': away,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'commence_time': commence,
            'bookie': bookie_key
        })

    odds_df = pd.DataFrame(rows)
    # generate a game_id based on teams for merging if necessary
    odds_df['game_id'] = odds_df['away_team'].astype(str) + '_vs_' + odds_df['home_team'].astype(str)

    return odds_df
