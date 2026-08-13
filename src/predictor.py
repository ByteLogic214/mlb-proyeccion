import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLBProjector:
    """Advanced MLB projection system with multi-factor analysis"""
    
    def __init__(self, data_path='data/today_games.csv'):
        self.data_path = data_path
        
        # League averages (2024 MLB)
        self.LEAGUE_AVG_ERA = 4.09
        self.LEAGUE_AVG_WHIP = 1.28
        self.LEAGUE_AVG_K9 = 9.2
        self.LEAGUE_AVG_BB9 = 2.8
        self.LEAGUE_AVG_AVG = 0.244
        self.LEAGUE_AVG_OBP = 0.314
        self.LEAGUE_AVG_SLG = 0.390
        self.LEAGUE_AVG_TOTALS = 8.8
        
        # Comprehensive park factors by venue
        self.PARK_FACTORS = {
            1: 1.02,    # Truist Stadium (ATL)
            2: 1.03,    # Oriole Park (BAL)
            3: 0.97,    # Fenway Park (BOS)
            4: 0.99,    # Great American Ball Park (CIN)
            5: 0.98,    # Guaranteed Rate Field (CWS)
            10: 1.02,   # Wrigley Field (CHC)
            11: 1.05,   # Progressive Field (CLE)
            12: 0.98,   # Comerica Park (DET)
            13: 1.08,   # Globe Life Field (TEX)
            14: 1.01,   # Rogers Centre (TOR)
            15: 1.04,   # Target Field (MIN)
            16: 0.96,   # Yankee Stadium (NYY)
            17: 0.95,   # Citi Field (NYM)
            18: 0.98,   # Citizens Bank Park (PHI)
            19: 1.05,   # PNC Park (PIT)
            20: 1.04,   # Busch Stadium (STL)
            21: 1.02,   # Petco Park (SD)
            22: 1.11,   # Dodger Stadium (LAD)
            25: 1.01,   # Oakland Coliseum (OAK)
            26: 1.06,   # Safeco Field (SEA)
            30: 1.04,   # Chase Field (ARI)
            33: 1.02,   # Miller Park (MIL)
            34: 1.03,   # Rangers Ballpark (NYY away)
            35: 1.07,   # Marlins Park (MIA)
            36: 0.95,   # Nationals Park (WSH)
            37: 1.10,   # Minute Maid Park (HOU)
            38: 1.02,   # Kauffman Stadium (KC)
            39: 1.09,   # Coors Field (COL)
            40: 1.04,   # Angel Stadium (LAA)
        }
        
        self.scaler = StandardScaler()

    def _load_data(self):
        """Load and validate game data with smart imputation"""
        try:
            df = pd.read_csv(self.data_path)
            
            if df.empty:
                print("No data available for projection.")
                return pd.DataFrame()
            
            # Fill missing pitcher stats with league averages
            pitcher_fill = {
                'away_sp_era': self.LEAGUE_AVG_ERA,
                'away_sp_whip': self.LEAGUE_AVG_WHIP,
                'away_sp_k9': self.LEAGUE_AVG_K9,
                'away_sp_bb9': self.LEAGUE_AVG_BB9,
                'away_sp_qs_pct': 0.35,
                'home_sp_era': self.LEAGUE_AVG_ERA,
                'home_sp_whip': self.LEAGUE_AVG_WHIP,
                'home_sp_k9': self.LEAGUE_AVG_K9,
                'home_sp_bb9': self.LEAGUE_AVG_BB9,
                'home_sp_qs_pct': 0.35,
            }
            
            team_fill = {
                'away_team_era': self.LEAGUE_AVG_ERA,
                'away_team_whip': self.LEAGUE_AVG_WHIP,
                'away_team_avg': self.LEAGUE_AVG_AVG,
                'away_team_obp': self.LEAGUE_AVG_OBP,
                'away_team_slg': self.LEAGUE_AVG_SLG,
                'away_recent_rpg': 4.4,
                'home_team_era': self.LEAGUE_AVG_ERA,
                'home_team_whip': self.LEAGUE_AVG_WHIP,
                'home_team_avg': self.LEAGUE_AVG_AVG,
                'home_team_obp': self.LEAGUE_AVG_OBP,
                'home_team_slg': self.LEAGUE_AVG_SLG,
                'home_recent_rpg': 4.4,
            }
            
            pitcher_fill.update(team_fill)
            df = df.fillna(pitcher_fill)
            
            # Validate critical columns exist
            required_cols = ['away_team', 'home_team', 'away_sp_era', 'home_sp_era', 'venue_id']
            if not all(col in df.columns for col in required_cols):
                print("Warning: Missing required columns")
                return df
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def _get_park_factor(self, venue_id):
        """Get accurate park factor for stadium"""
        return self.PARK_FACTORS.get(venue_id, 1.0)

    def _calculate_pitcher_rating(self, era, whip, k9, bb9, qs_pct):
        """
        Calculate comprehensive pitcher rating (0-100 scale)
        Lower ERA/WHIP is better, higher K9/BB9 ratio is better
        """
        # Normalize components (inverse for ERA/WHIP since lower is better)
        era_rating = (self.LEAGUE_AVG_ERA / max(era, 0.1)) * 50
        whip_rating = (self.LEAGUE_AVG_WHIP / max(whip, 0.1)) * 20
        k9_rating = (k9 / self.LEAGUE_AVG_K9) * 15
        bb9_penalty = (bb9 / self.LEAGUE_AVG_BB9) * 10
        qs_rating = (qs_pct * 5) if not np.isnan(qs_pct) else 0
        
        rating = era_rating + whip_rating + k9_rating - bb9_penalty + qs_rating
        return np.clip(rating, 0, 100)

    def _calculate_offense_rating(self, avg, obp, slg, recent_rpg):
        """
        Calculate team offensive rating
        Based on hitting metrics and recent performance
        """
        avg_rating = (avg / self.LEAGUE_AVG_AVG) * 30 if avg > 0 else 30
        obp_rating = (obp / self.LEAGUE_AVG_OBP) * 30 if obp > 0 else 30
        slg_rating = (slg / self.LEAGUE_AVG_SLG) * 25 if slg > 0 else 25
        recent_rating = (recent_rpg / 4.4) * 15 if recent_rpg > 0 else 15
        
        rating = avg_rating + obp_rating + slg_rating + recent_rating
        return np.clip(rating, 0, 100)

    def _calculate_defense_rating(self, team_era, team_whip):
        """
        Calculate team defensive rating
        Based on pitching staff performance
        """
        era_rating = (self.LEAGUE_AVG_ERA / max(team_era, 0.1)) * 50
        whip_rating = (self.LEAGUE_AVG_WHIP / max(team_whip, 0.1)) * 50
        rating = era_rating + whip_rating
        return np.clip(rating, 0, 100)

    def _calculate_win_probability(self, away_offense, away_defense, home_offense, home_defense):
        """
        Calculate win probability using advanced rating system
        Accounts for both offensive and defensive capabilities
        """
        # Home field advantage (~3.5%)
        home_advantage = 0.035
        
        # Combined strength metrics
        away_strength = (away_offense * 0.45) + (away_defense * 0.55)
        home_strength = (home_offense * 0.45) + (home_defense * 0.55)
        
        # Strength difference on normalized scale (-1 to 1)
        strength_diff = (home_strength - away_strength) / 100.0
        
        # Apply logistic function with home advantage
        home_prob = expit(strength_diff * 3.5) + home_advantage
        home_prob = np.clip(home_prob, 0.1, 0.9)  # Ensure reasonable probability
        
        return np.clip(home_prob, 0, 1), np.clip(1 - home_prob, 0, 1)

    def _calculate_total_runs(self, park_factor, pitcher_factor_away, pitcher_factor_home, 
                              away_offense, home_offense):
        """
        Calculate projected total runs using multiple factors
        """
        # Base on league average
        base_total = self.LEAGUE_AVG_TOTALS
        
        # Park factor adjustment
        adjusted = base_total * park_factor
        
        # Pitcher quality adjustment (average of both pitchers)
        pitcher_adj = (pitcher_factor_away + pitcher_factor_home) / 2
        adjusted *= pitcher_adj
        
        # Offensive quality factor (0.5 to 1.5 scale)
        avg_offense = (away_offense + home_offense) / 2 / 50.0
        adjusted *= np.clip(avg_offense, 0.5, 1.5)
        
        return np.clip(adjusted, 4.0, 15.0)  # Reasonable bounds

    def run_projections(self):
        """Generate advanced projections for all games"""
        df = self._load_data()
        if df.empty:
            return None

        projections = []
        for _, row in df.iterrows():
            try:
                # Calculate pitcher ratings
                away_sp_rating = self._calculate_pitcher_rating(
                    row.get('away_sp_era', self.LEAGUE_AVG_ERA),
                    row.get('away_sp_whip', self.LEAGUE_AVG_WHIP),
                    row.get('away_sp_k9', self.LEAGUE_AVG_K9),
                    row.get('away_sp_bb9', self.LEAGUE_AVG_BB9),
                    row.get('away_sp_qs_pct', 0.35)
                )
                
                home_sp_rating = self._calculate_pitcher_rating(
                    row.get('home_sp_era', self.LEAGUE_AVG_ERA),
                    row.get('home_sp_whip', self.LEAGUE_AVG_WHIP),
                    row.get('home_sp_k9', self.LEAGUE_AVG_K9),
                    row.get('home_sp_bb9', self.LEAGUE_AVG_BB9),
                    row.get('home_sp_qs_pct', 0.35)
                )
                
                # Calculate team ratings
                away_offense = self._calculate_offense_rating(
                    row.get('away_team_avg', self.LEAGUE_AVG_AVG),
                    row.get('away_team_obp', self.LEAGUE_AVG_OBP),
                    row.get('away_team_slg', self.LEAGUE_AVG_SLG),
                    row.get('away_recent_rpg', 4.4)
                )
                
                away_defense = self._calculate_defense_rating(
                    row.get('away_team_era', self.LEAGUE_AVG_ERA),
                    row.get('away_team_whip', self.LEAGUE_AVG_WHIP)
                )
                
                home_offense = self._calculate_offense_rating(
                    row.get('home_team_avg', self.LEAGUE_AVG_AVG),
                    row.get('home_team_obp', self.LEAGUE_AVG_OBP),
                    row.get('home_team_slg', self.LEAGUE_AVG_SLG),
                    row.get('home_recent_rpg', 4.4)
                )
                
                home_defense = self._calculate_defense_rating(
                    row.get('home_team_era', self.LEAGUE_AVG_ERA),
                    row.get('home_team_whip', self.LEAGUE_AVG_WHIP)
                )
                
                # Calculate win probabilities
                home_win_prob, away_win_prob = self._calculate_win_probability(
                    away_offense, away_defense, home_offense, home_defense
                )
                
                # Calculate total runs
                park_factor = self._get_park_factor(row.get('venue_id', 1))
                pitcher_factor_away = away_sp_rating / 100.0 * 1.2 + 0.8  # Scale to 0.8-2.0
                pitcher_factor_home = home_sp_rating / 100.0 * 1.2 + 0.8
                
                projected_total = self._calculate_total_runs(
                    park_factor, pitcher_factor_away, pitcher_factor_home,
                    away_offense, home_offense
                )
                
                # Calculate confidence score (0-1, based on data completeness)
                confidence = 1.0
                if np.isnan(row.get('away_sp_era')) or np.isnan(row.get('home_sp_era')):
                    confidence -= 0.15
                if np.isnan(row.get('away_team_avg')) or np.isnan(row.get('home_team_avg')):
                    confidence -= 0.10
                
                projections.append({
                    'game_id': row.get('game_id'),
                    'away_team': row.get('away_team'),
                    'home_team': row.get('home_team'),
                    'away_sp': row.get('away_sp_name', 'TBD'),
                    'away_sp_rating': round(away_sp_rating, 1),
                    'home_sp': row.get('home_sp_name', 'TBD'),
                    'home_sp_rating': round(home_sp_rating, 1),
                    'away_offense_rating': round(away_offense, 1),
                    'away_defense_rating': round(away_defense, 1),
                    'home_offense_rating': round(home_offense, 1),
                    'home_defense_rating': round(home_defense, 1),
                    'prob_home_win': round(home_win_prob, 3),
                    'prob_away_win': round(away_win_prob, 3),
                    'projected_total': round(projected_total, 2),
                    'confidence': round(confidence, 2),
                    'venue_id': row.get('venue_id')
                })
            except Exception as e:
                print(f"Error processing projection for game {row.get('game_id')}: {e}")
                continue

        return pd.DataFrame(projections) if projections else None

if __name__ == "__main__":
    projector = MLBProjector()
    results = projector.run_projections()
    if results is not None and not results.empty:
        results.to_csv('data/projections.csv', index=False)
        print("\nProjections saved to data/projections.csv")
        print(f"\nGenerated {len(results)} game projections")
    else:
        print("No projections generated")
