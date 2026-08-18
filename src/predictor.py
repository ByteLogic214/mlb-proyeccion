import numpy as np
import pandas as pd


class MLBProjector:
    """Advanced MLB projection system with multi-factor analysis.

    Model v2 (precision-focused rebuild):
    - Ratings centrados en 50 (media de liga) con ~15 puntos por desviación estándar.
      (La versión anterior usaba media 100 con clip a 100, desperdiciando media escala.)
    - Probabilidad de victoria vía log5 (Bill James) que combina fuerza de equipo
      Y calidad del abridor, con ventaja de local (~54% histórico en MLB).
      (La versión anterior ignoraba por completo a los pitchers abridores.)
    - Park factors con venue IDs reales de MLB StatsAPI y valores calibrados
      multi-temporada. (La versión anterior usaba IDs incorrectos, por lo que
      casi todos los juegos caían en el fallback 1.0.)
    - Total de carreras con dirección corregida: mejores abridores y defensas
      REDUCEN las carreras proyectadas. (La versión anterior tenía el signo
      invertido: mejor pitcher => más carreras.)
    """

    # Ventaja histórica del equipo local en MLB (~54% de victorias)
    HOME_WIN_PCT = 0.54

    def __init__(self, data_path='data/today_games.csv'):
        self.data_path = data_path

        # League averages (MLB 2024)
        self.LEAGUE_AVG_ERA = 4.09
        self.LEAGUE_AVG_WHIP = 1.28
        self.LEAGUE_AVG_K9 = 9.2
        self.LEAGUE_AVG_BB9 = 2.8
        self.LEAGUE_AVG_AVG = 0.244
        self.LEAGUE_AVG_OBP = 0.314
        self.LEAGUE_AVG_SLG = 0.390
        self.LEAGUE_AVG_TOTALS = 8.8

        # Rango realista de totales (permite Coors y duelos de pitcheo)
        self.MIN_TOTAL = 5.5
        self.MAX_TOTAL = 13.5

        # Park factors con venue IDs reales de MLB StatsAPI.
        # Valores multi-temporada (carreras), 1.00 = neutral.
        self.PARK_FACTORS = {
            1: 0.97,     # Angel Stadium (LAA)
            2: 0.99,     # Oriole Park at Camden Yards (BAL)
            3: 1.08,     # Fenway Park (BOS)
            4: 1.04,     # Guaranteed Rate Field (CWS)
            5: 1.00,     # Progressive Field (CLE)
            7: 1.02,     # Kauffman Stadium (KC)
            10: 0.94,    # Oakland Coliseum (OAK)
            12: 0.93,    # Tropicana Field (TB)
            14: 1.00,    # Rogers Centre (TOR)
            15: 1.01,    # Chase Field (ARI)
            17: 1.01,    # Wrigley Field (CHC)
            19: 1.32,    # Coors Field (COL) — el parque más bateador de MLB
            22: 0.95,    # Dodger Stadium (LAD)
            31: 0.94,    # PNC Park (PIT)
            32: 1.00,    # American Family Field (MIL)
            680: 0.90,   # T-Mobile Park (SEA) — el más pitcher-friendly
            2392: 1.01,  # Minute Maid Park (HOU)
            2394: 0.95,  # Comerica Park (DET)
            2395: 0.90,  # Oracle Park (SF)
            2602: 1.09,  # Great American Ball Park (CIN)
            2680: 0.93,  # Petco Park (SD)
            2681: 1.02,  # Citizens Bank Park (PHI)
            2889: 0.95,  # Busch Stadium (STL)
            3289: 0.94,  # Citi Field (NYM)
            3309: 0.99,  # Nationals Park (WSH)
            3312: 1.00,  # Target Field (MIN)
            3313: 1.03,  # Yankee Stadium (NYY)
            4169: 0.94,  # loanDepot park (MIA)
            4705: 1.00,  # Truist Park (ATL)
            5325: 1.00,  # Globe Life Field (TEX)
        }

        # Distribuciones a nivel de liga para z-scores (se estiman del CSV del día)
        self.offense_means = None
        self.offense_stds = None
        self.defense_means = None
        self.defense_stds = None

    # ------------------------------------------------------------------
    # Utilidades estadísticas
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _log5(prob_a, prob_b):
        """Log5 de Bill James: P(A vence a B) dadas las probabilidades de victoria
        esperadas de cada equipo contra un rival promedio."""
        a = float(np.clip(prob_a, 0.01, 0.99))
        b = float(np.clip(prob_b, 0.01, 0.99))
        denom = a + b - 2.0 * a * b
        if abs(denom) < 1e-9:
            return 0.5
        return (a - a * b) / denom

    # ------------------------------------------------------------------
    # Carga y validación de datos
    # ------------------------------------------------------------------
    def _load_data(self):
        """Load and validate game data with smart imputation"""
        try:
            df = pd.read_csv(self.data_path)

            if df.empty:
                print("No data available for projection.")
                return pd.DataFrame()

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

            required_cols = ['away_team', 'home_team', 'away_sp_era', 'home_sp_era', 'venue_id']
            if not all(col in df.columns for col in required_cols):
                print("Warning: Missing required columns")
                return df

            # Distribuciones de liga aproximadas combinando filas away/home
            try:
                team_offense = pd.DataFrame()
                team_offense['avg'] = pd.concat([df['away_team_avg'], df['home_team_avg']], ignore_index=True)
                team_offense['obp'] = pd.concat([df['away_team_obp'], df['home_team_obp']], ignore_index=True)
                team_offense['slg'] = pd.concat([df['away_team_slg'], df['home_team_slg']], ignore_index=True)
                team_offense['recent_rpg'] = pd.concat([df['away_recent_rpg'], df['home_recent_rpg']], ignore_index=True)

                team_defense = pd.DataFrame()
                team_defense['era'] = pd.concat([df['away_team_era'], df['home_team_era']], ignore_index=True)
                team_defense['whip'] = pd.concat([df['away_team_whip'], df['home_team_whip']], ignore_index=True)

                self.offense_means = team_offense.mean().to_dict()
                self.offense_stds = team_offense.std(ddof=0).replace(0, 1e-6).to_dict()
                self.defense_means = team_defense.mean().to_dict()
                self.defense_stds = team_defense.std(ddof=0).replace(0, 1e-6).to_dict()
            except Exception:
                self.offense_means = {'avg': self.LEAGUE_AVG_AVG, 'obp': self.LEAGUE_AVG_OBP,
                                      'slg': self.LEAGUE_AVG_SLG, 'recent_rpg': 4.4}
                self.offense_stds = {'avg': 0.02, 'obp': 0.02, 'slg': 0.05, 'recent_rpg': 0.5}
                self.defense_means = {'era': self.LEAGUE_AVG_ERA, 'whip': self.LEAGUE_AVG_WHIP}
                self.defense_stds = {'era': 0.5, 'whip': 0.1}

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def _get_park_factor(self, venue_id):
        """Park factor por venue ID real de MLB StatsAPI (1.0 si es desconocido)"""
        try:
            return self.PARK_FACTORS.get(int(venue_id), 1.0)
        except (TypeError, ValueError):
            return 1.0

    # ------------------------------------------------------------------
    # Ratings (escala 0-100, media de liga = 50, ~15 pts por desviación)
    # ------------------------------------------------------------------
    def _calculate_pitcher_rating(self, era, whip, k9, bb9, qs_pct):
        """Rating de abridor (0-100). Media de liga = 50; un as típico ≈ 65-75."""
        z_era = (self.LEAGUE_AVG_ERA - era) / 1.00
        z_whip = (self.LEAGUE_AVG_WHIP - whip) / 0.12
        z_kbb = ((k9 - bb9) - (self.LEAGUE_AVG_K9 - self.LEAGUE_AVG_BB9)) / 2.5
        z_qs = (qs_pct - 0.35) / 0.15 if not pd.isna(qs_pct) else 0.0

        combined_z = (z_era * 0.35) + (z_whip * 0.25) + (z_kbb * 0.25) + (z_qs * 0.15)
        return float(np.clip(50.0 + combined_z * 15.0, 5.0, 95.0))

    def _calculate_offense_rating(self, avg, obp, slg, recent_rpg):
        """Rating ofensivo (0-100) vía z-scores contra la distribución de liga."""
        means = self.offense_means or {'avg': self.LEAGUE_AVG_AVG, 'obp': self.LEAGUE_AVG_OBP,
                                       'slg': self.LEAGUE_AVG_SLG, 'recent_rpg': 4.4}
        stds = self.offense_stds or {'avg': 0.02, 'obp': 0.02, 'slg': 0.05, 'recent_rpg': 0.5}

        z_avg = (avg - means['avg']) / stds['avg']
        z_obp = (obp - means['obp']) / stds['obp']
        z_slg = (slg - means['slg']) / stds['slg']
        z_recent = (recent_rpg - means['recent_rpg']) / stds['recent_rpg']

        combined_z = (z_avg * 0.25) + (z_obp * 0.30) + (z_slg * 0.30) + (z_recent * 0.15)
        return float(np.clip(50.0 + combined_z * 15.0, 5.0, 95.0))

    def _calculate_defense_rating(self, team_era, team_whip):
        """Rating defensivo (0-100). Menor ERA/WHIP => mejor defensa."""
        means = self.defense_means or {'era': self.LEAGUE_AVG_ERA, 'whip': self.LEAGUE_AVG_WHIP}
        stds = self.defense_stds or {'era': 0.5, 'whip': 0.1}

        z_era = (means['era'] - team_era) / stds['era']
        z_whip = (means['whip'] - team_whip) / stds['whip']

        combined_z = (z_era * 0.6) + (z_whip * 0.4)
        return float(np.clip(50.0 + combined_z * 15.0, 5.0, 95.0))

    # ------------------------------------------------------------------
    # Probabilidad de victoria
    # ------------------------------------------------------------------
    def _expected_win_pct(self, offense, defense, sp_rating):
        """Win% esperado del equipo contra un rival promedio, combinando
        fuerza de equipo (ofensiva + defensa) y calidad del abridor del día."""
        strength = (offense * 0.45) + (defense * 0.55)
        # La desviación estándar de win% entre equipos MLB ≈ 0.070
        win_pct = 0.5 + (strength - 50.0) * 0.0070
        # Un as (rating ~80) mueve el win% del día ~+0.09; un quinto abridor lo hunde
        win_pct += (sp_rating - 50.0) * 0.0030
        return float(np.clip(win_pct, 0.28, 0.72))

    def _calculate_win_probability(self, away_offense, away_defense, home_offense,
                                   home_defense, away_sp_rating, home_sp_rating):
        """P(victoria local) vía log5 equipo+abridor, con ventaja de local."""
        away_wpct = self._expected_win_pct(away_offense, away_defense, away_sp_rating)
        home_wpct = self._expected_win_pct(home_offense, home_defense, home_sp_rating)

        neutral_prob = self._log5(home_wpct, away_wpct)

        # Ventaja de local aplicada en espacio logit (~54% histórico)
        home_logit = self._logit(neutral_prob) + self._logit(self.HOME_WIN_PCT)
        home_prob = float(self._sigmoid(home_logit))

        # Rango realista para moneylines MLB
        home_prob = float(np.clip(home_prob, 0.20, 0.80))
        return home_prob, float(1.0 - home_prob)

    # ------------------------------------------------------------------
    # Total de carreras
    # ------------------------------------------------------------------
    def _sp_run_factor(self, era, whip):
        """Factor de carreras permitidas del abridor (1.0 = pitcher promedio)."""
        return 0.6 * (era / self.LEAGUE_AVG_ERA) + 0.4 * (whip / self.LEAGUE_AVG_WHIP)

    def _defense_run_factor(self, defense_rating):
        """Factor del resto del staff/defensa (1.0 = promedio)."""
        return 1.0 + (50.0 - defense_rating) / 50.0 * 0.25

    def _offense_run_factor(self, offense_rating):
        """Factor ofensivo (1.0 = ofensiva promedio)."""
        return 1.0 + (offense_rating - 50.0) / 50.0 * 0.50

    def _calculate_total_runs(self, park_factor, away_offense, home_offense,
                              away_defense, home_defense,
                              away_sp_era, away_sp_whip, home_sp_era, home_sp_whip):
        """Total proyectado = carreras esperadas de cada equipo.

        Cada equipo anota según: base de liga × parque × su ofensiva ×
        calidad del abridor rival × defensa rival. Mejor pitcheo/defensa
        => MENOS carreras (dirección corregida respecto a la versión anterior).
        """
        half = self.LEAGUE_AVG_TOTALS / 2.0

        away_env = (0.6 * self._sp_run_factor(home_sp_era, home_sp_whip)
                    + 0.4 * self._defense_run_factor(home_defense))
        home_env = (0.6 * self._sp_run_factor(away_sp_era, away_sp_whip)
                    + 0.4 * self._defense_run_factor(away_defense))

        away_runs = half * park_factor * self._offense_run_factor(away_offense) * away_env
        home_runs = half * park_factor * self._offense_run_factor(home_offense) * home_env * 1.03

        return float(np.clip(away_runs + home_runs, self.MIN_TOTAL, self.MAX_TOTAL))

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------
    def run_projections(self):
        """Generate advanced projections for all games"""
        df = self._load_data()
        if df.empty:
            return None

        projections = []
        for _, row in df.iterrows():
            try:
                away_sp_era = row.get('away_sp_era', self.LEAGUE_AVG_ERA)
                away_sp_whip = row.get('away_sp_whip', self.LEAGUE_AVG_WHIP)
                home_sp_era = row.get('home_sp_era', self.LEAGUE_AVG_ERA)
                home_sp_whip = row.get('home_sp_whip', self.LEAGUE_AVG_WHIP)

                away_sp_rating = self._calculate_pitcher_rating(
                    away_sp_era, away_sp_whip,
                    row.get('away_sp_k9', self.LEAGUE_AVG_K9),
                    row.get('away_sp_bb9', self.LEAGUE_AVG_BB9),
                    row.get('away_sp_qs_pct', 0.35)
                )
                home_sp_rating = self._calculate_pitcher_rating(
                    home_sp_era, home_sp_whip,
                    row.get('home_sp_k9', self.LEAGUE_AVG_K9),
                    row.get('home_sp_bb9', self.LEAGUE_AVG_BB9),
                    row.get('home_sp_qs_pct', 0.35)
                )

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

                # Win probability ahora SÍ incluye a los abridores
                home_win_prob, away_win_prob = self._calculate_win_probability(
                    away_offense, away_defense, home_offense, home_defense,
                    away_sp_rating, home_sp_rating
                )

                park_factor = self._get_park_factor(row.get('venue_id', 1))
                projected_total = self._calculate_total_runs(
                    park_factor, away_offense, home_offense, away_defense, home_defense,
                    away_sp_era, away_sp_whip, home_sp_era, home_sp_whip
                )

                # Confianza según completitud de datos
                confidence = 1.0
                if pd.isna(row.get('away_sp_era')) or pd.isna(row.get('home_sp_era')):
                    confidence -= 0.15
                if pd.isna(row.get('away_team_avg')) or pd.isna(row.get('home_team_avg')):
                    confidence -= 0.10

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
