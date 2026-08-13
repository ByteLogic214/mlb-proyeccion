import pandas as pd
import numpy as np
from scipy.special import expit  # Sigmoid function for probabilities

class MLBProjector:
    def __init__(self, data_path='data/today_games.csv'):
        self.data_path = data_path
        # League averages (constants for normalization)
        self.LEAGUE_AVG_ERA = 4.30
        self.LEAGUE_AVG_TOTALS = 9.0
        
        # Park Factors: 1.0 is neutral. > 1.0 is hitter-friendly, < 1.0 is pitcher-friendly.
        # Mapping common venue_ids (Example IDs - in production, use a full mapping)
        self.PARK_FACTORS = {
            166: 1.15,  # Coors Field (Extreme hitter friendly)
            167: 1.05,  # Fenway Park
            168: 0.95,  # Petco Park (Pitcher friendly)
            # Default for unknown venues will be 1.0
        }

    def _load_data(self):
        """Loads the data and handles missing values with league averages."""
        try:
            df = pd.read_csv(self.data_path)
            # Fill NaNs with league averages to prevent calculation errors
            fill_values = {
                'away_sp_era': self.LEAGUE_AVG_ERA,
                'away_sp_whip': 1.30,
                'away_team_era': self.LEAGUE_AVG_ERA,
                'home_sp_era': self.LEAGUE_AVG_ERA,
                'home_sp_whip': 1.30,
                'home_team_era': self.LEAGUE_AVG_ERA
            }
            return df.fillna(value=fill_values)
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found.")
            return pd.DataFrame()

    def _get_park_factor(self, venue_id):
        """Returns the park factor for a given venue ID."""
        return self.PARK_FACTORS.get(venue_id, 1.0)

    def _calculate_pitching_score(self, sp_era, bullpen_era):
        """
        Calculates a weighted pitching score. 
        Lower score is better (closer to 0).
        Weight: 70% Starter, 30% Bullpen.
        """
        # We normalize against league average: (League_ERA / Team_ERA)
        # A score > 1 means better than average, < 1 means worse.
        sp_component = self.LEAGUE_AVG_ERA / sp_era
        bp_component = self.LEAGUE_AVG_ERA / bullpen_era
        return (0.7 * sp_component) + (0.3 * bp_component)

    def run_projections(self):
        """Main engine to run Moneyline and Totals projections."""
        df = self._load_data()
        if df.empty:
            return None

        projections = []

        for _, row in df.iterrows():
            # 1. Calculate Pitching Strength for both teams
            away_strength = self._calculate_pitching_score(row['away_sp_era'], row['away_team_era'])
            home_strength = self._calculate_pitching_score(row['home_sp_era'], row['home_team_era'])

            # 2. Moneyline Projection (Win Probability)
            # We use the difference in strength passed through a sigmoid function
            # We scale the difference to make the probability curve realistic
            strength_diff = home_strength - away_strength
            home_win_prob = expit(strength_diff * 5) # 5 is a scaling factor for sensitivity
            away_win_prob = 1 - home_win_prob

            # 3. Totals Projection (Over/Under)
            park_factor = self._get_park_factor(row['venue_id'])
            
            # Base runs adjusted by pitching efficiency and park factor
            # If pitchers are better than average, they reduce the total runs.
            pitching_adjustment = (away_strength + home_strength) / 2
            projected_total = self.LEAGUE_AVG_TOTALS * park_factor * pitching_adjustment

            projections.append({
                'game_id': row['game_id'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'away_sp': row['away_sp_name'],
                'home_sp': row['home_sp_name'],
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
        print("--- MLB DAILY PROJECTIONS ---")
        print(results.to_string(index=False))
        # Save results for the notifier
        results.to_csv('data/projections.csv', index=False)
    else:
        print("No projections generated.")
