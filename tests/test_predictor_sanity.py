"""Sanity tests for the MLB projection model (v2).

These tests lock in the model's core directional properties so future
changes can't silently break them:
- League-average inputs => ratings near 50 and ~54% home win probability
- Better starting pitcher => higher win probability AND lower projected total
- Real StatsAPI venue IDs resolve to sensible park factors
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import MLBProjector  # noqa: E402


def _projector():
    return MLBProjector()


def test_league_average_pitcher_rating_is_50():
    p = _projector()
    rating = p._calculate_pitcher_rating(4.09, 1.28, 9.2, 2.8, 0.35)
    assert 45.0 <= rating <= 55.0


def test_ace_rates_much_higher_than_bad_pitcher():
    p = _projector()
    ace = p._calculate_pitcher_rating(2.60, 1.02, 11.0, 2.0, 0.70)
    bad = p._calculate_pitcher_rating(5.60, 1.55, 7.0, 4.0, 0.10)
    assert ace > 65.0
    assert bad < 35.0
    assert ace > bad


def test_even_matchup_gives_home_field_edge():
    p = _projector()
    home_prob, away_prob = p._calculate_win_probability(50, 50, 50, 50, 50, 50)
    assert 0.52 <= home_prob <= 0.56
    assert abs(home_prob + away_prob - 1.0) < 1e-9


def test_better_starting_pitcher_raises_win_probability():
    p = _projector()
    base_home, _ = p._calculate_win_probability(50, 50, 50, 50, 50, 50)
    ace_home, _ = p._calculate_win_probability(50, 50, 50, 50, 50, 80)
    assert ace_home > base_home


def test_better_pitching_lowers_projected_total():
    """Direction bug fix: better starters must REDUCE projected runs."""
    p = _projector()
    total_avg = p._calculate_total_runs(1.0, 50, 50, 50, 50, 4.09, 1.28, 4.09, 1.28)
    total_ace = p._calculate_total_runs(1.0, 50, 50, 50, 50, 4.09, 1.28, 2.60, 1.02)
    assert total_ace < total_avg


def test_hitter_friendly_park_raises_projected_total():
    p = _projector()
    total_neutral = p._calculate_total_runs(1.00, 50, 50, 50, 50, 4.09, 1.28, 4.09, 1.28)
    total_coors = p._calculate_total_runs(1.32, 50, 50, 50, 50, 4.09, 1.28, 4.09, 1.28)
    assert total_coors > total_neutral


def test_park_factors_use_real_statsapi_venue_ids():
    p = _projector()
    # Coors Field = 19 (most hitter-friendly), Yankee Stadium = 3313
    assert p._get_park_factor(19) == max(p.PARK_FACTORS.values())
    assert p._get_park_factor(19) > 1.2
    assert p._get_park_factor(3313) > 1.0
    assert p._get_park_factor(99999) == 1.0  # fallback for unknown venues


def test_win_probability_stays_in_realistic_range():
    p = _projector()
    home_prob, _ = p._calculate_win_probability(20, 20, 80, 80, 30, 85)
    assert 0.20 <= home_prob <= 0.80


def test_projected_total_stays_in_bounds():
    p = _projector()
    low = p._calculate_total_runs(0.90, 20, 20, 90, 90, 2.0, 0.95, 2.0, 0.95)
    high = p._calculate_total_runs(1.32, 90, 90, 10, 10, 6.5, 1.7, 6.5, 1.7)
    assert low >= p.MIN_TOTAL
    assert high <= p.MAX_TOTAL
