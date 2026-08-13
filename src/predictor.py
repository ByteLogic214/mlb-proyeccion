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
        
        # Reasonable total runs bounds (more realistic)
        self.MIN_TOTAL = 7.0
        self.MAX_TOTAL = 11.5
        
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

        # placeholders for season-level means/stds used for z-score normalisation
        self.offense_means = None
        self.offense_stds = None
        self.defense_means = None
        self.defense_stds = None

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

            # Build season-level distributions for normalization
            # Combine away/home team offensive rows to approximate team-level distribution
            try:
                team_offense = pd.DataFrame()
                team_offense['avg'] = pd.concat([df['away_team_avg'], df['home_team_avg']], ignore_index=True)
                team_offense['obp'] = pd.concat([df['away_team_obp'], df['home_team_obp']], ignore_index=True)
                team_offense['slg'] = pd.concat([df['away_team_slg'], df['home_team_slg']], ignore_index=True)
                team_offense['recent_rpg'] = pd.concat([df['away_recent_rpg'], df['home_recent_rpg']], ignore_index=True)

                # defensive metrics
                team_defense = pd.DataFrame()
                team_defense['era'] = pd.concat([df['away_team_era'], df['home_team_era']], ignore_index=True)
                team_defense['whip'] = pd.concat([df['away_team_whip'], df['home_team_whip']], ignore_index=True)

                # compute means/stds with safe fallbacks
                self.offense_means = team_offense.mean().to_dict()
                self.offense_stds = team_offense.std(ddof=0).replace(0, 1e-6).to_dict()

                self.defense_means = team_defense.mean().to_dict()
                self.defense_stds = team_defense.std(ddof=0).replace(0, 1e-6).to_dict()
            except Exception:
                # if any of the columns are missing or error occurs, set defaults to league values
                self.offense_means = {'avg': self.LEAGUE_AVG_AVG, 'obp': self.LEAGUE_AVG_OBP, 'slg': self.LEAGUE_AVG_SLG, 'recent_rpg': 4.4}
                self.offense_stds = {'avg': 0.02, 'obp': 0.02, 'slg': 0.05, 'recent_rpg': 0.5}
                self.defense_means = {'era': self.LEAGUE_AVG_ERA, 'whip': self.LEAGUE_AVG_WHIP}
                self.defense_stds = {'era': 0.5, 'whip': 0.1}
            
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
        k9_rating = (k9 / max(self.LEAGUE_AVG_K9, 1e-6)) * 15
        bb9_penalty = (bb9 / max(self.LEAGUE_AVG_BB9, 1e-6)) * 10
        qs_rating = (qs_pct * 5) if not pd.isna(qs_pct) else 0
        
        rating = era_rating + whip_rating + k9_rating - bb9_penalty + qs_rating
        return float(np.clip(rating, 0, 100))

    def _calculate_offense_rating(self, avg, obp, slg, recent_rpg):
        """
        Calculate team offensive rating using z-score normalization mapped to 0-100
        Centers around 100 for average teams and gives a Gaussian-like distribution
        """
        means = self.offense_means or {'avg': self.LEAGUE_AVG_AVG, 'obp': self.LEAGUE_AVG_OBP, 'slg': self.LEAGUE_AVG_SLG, 'recent_rpg': 4.4}
        stds = self.offense_stds or {'avg': 0.02, 'obp': 0.02, 'slg': 0.05, 'recent_rpg': 0.5}

        # safe z-scores
        z_avg = (avg - means['avg']) / stds['avg']
        z_obp = (obp - means['obp']) / stds['obp']
        z_slg = (slg - means['slg']) / stds['slg']
        z_recent = (recent_rpg - means['recent_rpg']) / stds['recent_rpg']

        # weighted combination (weights tuned so typical z's map sensibly)
        combined_z = (z_avg * 0.25) + (z_obp * 0.30) + (z_slg * 0.30) + (z_recent * 0.15)

        # map to 0-100 with 100 as mean and 15 points per std (approx Gaussian)
        score = 100.0 + (combined_z * 15.0)
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_defense_rating(self, team_era, team_whip):
        """
        Calculate team defensive rating using z-scores.
        Lower ERA/WHIP is better so invert those z-scores.
        """
        means = self.defense_means or {'era': self.LEAGUE_AVG_ERA, 'whip': self.LEAGUE_AVG_WHIP}
        stds = self.defense_stds or {'era': 0.5, 'whip': 0.1}

        # For defense, lower era/whip => positive z
        z_era = (means['era'] - team_era) / stds['era']
        z_whip = (means['whip'] - team_whip) / stds['whip']

        combined_z = (z_era * 0.6) + (z_whip * 0.4)
        score = 100.0 + (combined_z * 15.0)
        return float(np.clip(score, 0.0, 100.0))

    def _calculate_win_probability(self, away_offense, away_defense, home_offense, home_defense):
        """
        Calculate win probability using advanced rating system
        Increases sensitivity so mismatches produce wider probabilities.
        """
        # Home field advantage (~3.0%) applied as multiplier in logit space
        home_advantage = 0.03
        
        # Combined strength metrics
        away_strength = (away_offense * 0.45) + (away_defense * 0.55)
        home_strength = (home_offense * 0.45) + (home_defense * 0.55)
        
        # Strength difference scaled to roughly [-3, 3]
        strength_diff = (home_strength - away_strength) / 100.0 * 6.0
        
        # Apply logistic function with stronger slope to increase variance
        home_logit = strength_diff + (home_advantage * 6.0)
        home_prob = expit(home_logit)
        # Allow full probability range but avoid absolute extremes
        home_prob = np.clip(home_prob, 0.02, 0.98)
        
        return float(np.clip(home_prob, 0, 1)), float(np.clip(1 - home_prob, 0, 1))

    def _calculate_total_runs(self, park_factor, pitcher_factor_away, pitcher_factor_home, 
                              away_offense, home_offense, away_sp_era=None, home_sp_era=None):
        """
        Calculate projected total runs using multiple factors
        Incorporates SP ERA average (proxy for xFIP if not available), park, and team offense strength.
        """
        base_total = self.LEAGUE_AVG_TOTALS

        # Park factor
        adjusted = base_total * park_factor

        # Pitcher quality: use ERA average as a simple proxy. Lower ERA -> fewer runs
        try:
            sp_era_avg = np.nanmean([float(x) for x in [away_sp_era, home_sp_era] if x is not None and not pd.isna(x)])
        except Exception:
            sp_era_avg = self.LEAGUE_AVG_ERA

        # translate ERA into a multiplier centered near 1.0
        # better pitchers (era < league) reduce runs, worse increase
        era_diff = (sp_era_avg - self.LEAGUE_AVG_ERA) / max(self.LEAGUE_AVG_ERA, 1e-6)
        pitcher_era_scaler = 1.0 + (era_diff * 0.35)  # sensitivity factor

        # combine with supplied pitcher_factor (from pitcher ratings)
        pitcher_adj = ((pitcher_factor_away + pitcher_factor_home) / 2.0) * pitcher_era_scaler

        adjusted *= pitcher_adj

        # Offensive quality factor (use offense ratings on 0-100 mapped to ~0.8-1.2)
        avg_offense_rating = (away_offense + home_offense) / 2.0
        offense_scaler = 0.8 + (avg_offense_rating / 100.0) * 0.4  # maps 0->0.8, 100->1.2
        adjusted *= offense_scaler

        # Final realistic bounds
        return float(np.clip(adjusted, self.MIN_TOTAL, self.MAX_TOTAL))

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
                
                # Calculate team ratings using z-score normalization
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
                
                # Calculate win probabilities (more sensitive to mismatches)
                home_win_prob, away_win_prob = self._calculate_win_probability(
                    away_offense, away_defense, home_offense, home_defense
                )
                
                # Calculate total runs
                park_factor = self._get_park_factor(row.get('venue_id', 1))
                pitcher_factor_away = away_sp_rating / 100.0 * 1.2 + 0.8  # Scale to 0.8-2.0
                pitcher_factor_home = home_sp_rating / 100.0 * 1.2 + 0.8
                
                projected_total = self._calculate_total_runs(
                    park_factor, pitcher_factor_away, pitcher_factor_home,
                    away_offense, home_offense,
                    away_sp_era=row.get('away_sp_era', np.nan),
                    home_sp_era=row.get('home_sp_era', np.nan)
                )
                
                # Calculate confidence score (0-1, based on data completeness)
                confidence = 1.0
                if pd.isna(row.get('away_sp_era')) or pd.isna(row.get('home_sp_era')):
                    confidence -= 0.15
                if pd.isna(row.get('away_team_avg')) or pd.isna(row.get('home_team_avg')):
                    confidence -= 0.10

                # Penalize if pitchers are unresolved (bullpen day / TBD)
                away_name = row.get('away_sp_name', '') or ''
                home_name = row.get('home_sp_name', '') or ''
                if 'Bullpen' in away_name or away_name.strip().upper() in ['TBD', 'UNKNOWN', '']:
                    confidence -= 0.10
                if 'Bullpen' in home_name or home_name.strip().upper() in ['TBD', 'UNKNOWN', '']:
                    confidence -= 0.10

                confidence = float(np.clip(confidence, 0.0, 1.0))
                
                projections.append({
                    'game_id': row.get('game_id'),
                    'away_team': row.get('away_team'),
                    'home_team': row.get('home_team'),
                    'away_sp': row.get('away_sp_name', 'TBD') if row.get('away_sp_name') not in [None, ''] else 'TBD',
                    'away_sp_rating': round(away_sp_rating, 1),
                    'home_sp': row.get('home_sp_name', 'TBD') if row.get('home_sp_name') not in [None, ''] else 'TBD',
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
