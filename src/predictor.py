import pandas as pd
import numpy as np
from scipy.special import expit

class MLBProjector:
    def __init__(self, data_path='data/today_games.csv'):
        self.data_path = data_path
        self.LEAGUE_AVG_ERA = 4.30
        self.LEAGUE_AVG_TOTALS = 9.0
        self.PARK_FACTORS = {166: 1.15, 167: 1.05, 168: 0.95}

    def _load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            fill_values = {
                'away_sp_era': self.LEAGUE_AVG_ERA,
                'away_sp_whip': 1.30,
                'away_team_era': self.LEAGUE_AVG_ERA,
                'home_sp_era': self.LEAGUE_AVG_ERA,
                'home_sp_whip': 1.30,
                'home_team_era': self.LEAGUE_AVG_ERA
            }
            return df.fillna(value=fill_values)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def _get_park_factor(self, venue_id):
        return self.PARK_FACTORS.get(venue_id, 1.0)

    def _calculate_pitching_score(self, sp_era, bullpen_era):
        sp_component = self.LEAGUE_AVG_ERA / sp_era
        bp_component = self.LEAGUE_AVG_ERA / bullpen_era
        return (0.7 * sp_component) + (0.3 * bp_component)

    def run_projections(self):
        df = self._load_data()
        if df.empty:
            return None

        projections = []
        for _, row in df.iterrows():
            away_strength = self._calculate_pitching_score(row['away_sp_era'], row['away_team_era'])
            home_strength = self._calculate_pitching_score(row['home_sp_era'], row['home_team_era'])

            strength_diff = home_strength - away_strength
            home_win_prob = expit(strength_diff * 5)
            away_win_prob = 1 - home_win_prob

            park_factor = self._get_park_factor(row['venue_id'])
            pitching_adjustment = (away_strength + home_strength) / 2
            projected_total = self.LEAGUE_AVG_TOTALS * park_factor * pitching_adjustment

            # FIX: Included ERA columns in the output dictionary so main.py can access them
            projections.append({
                'game_id': row['game_id'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'away_sp': row['away_sp_name'],
                'away_sp_era': row['away_sp_era'],
                'home_sp': row['home_sp_name'],
                'home_sp_era': row['home_sp_era'],
                'prob_home_win': round(home_win_prob, 3),
                'prob_away_win': round(away_win_prob, 3),
                'projected_total': round(projected_total, 2),
                'venue_id': row['venue_id']
            })

        return pd.DataFrame(projections)

if __name__ == "__main__":
    projector = MLBProjector()
    results = projector.run_projections()
    if results is not None:
        results.to_csv('data/projections.csv', index=False)
        print("Projections saved.")
