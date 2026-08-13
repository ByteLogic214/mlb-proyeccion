import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os

class MLBDataFetcher:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.current_season = datetime.now().year
        self.output_path = "data/today_games.csv"
        
        # Ensure directory exists
        os.makedirs("data", exist_ok=True)

    def _get_json(self, endpoint, params=None):
        """Helper method to handle API requests."""
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            return None

    def get_pitcher_stats(self, player_id):
        """Fetches ERA, WHIP, W, and L for a specific pitcher."""
        if not player_id:
            return {"era": np.nan, "whip": np.nan, "wins": 0, "losses": 0}
        
        stats_data = self._get_json(f"people/{player_id}", {"season": self.current_season})
        
        # Default values
        pitcher_stats = {"era": np.nan, "whip": np.nan, "wins": 0, "losses": 0}
        
        try:
            # Navigate through the nested JSON structure
            # MLB API returns stats in a list under 'stats'
            if stats_data and 'stats' in stats_data and len(stats_data['stats']) > 0:
                stat_group = stats_data['stats'][0]['stat']
                
                pitcher_stats["era"] = float(stat_group.get('era', np.nan))
                pitcher_stats["whip"] = float(stat_group.get('whip', np.nan))
                pitcher_stats["wins"] = int(stat_group.get('wins', 0))
                pitcher_stats["losses"] = int(stat_group.get('losses', 0))
        except (KeyError, ValueError, IndexError) as e:
            print(f"Warning: Could not parse stats for player {player_id}: {e}")
            
        return pitcher_stats

    def get_team_bullpen_stats(self, team_id):
        """Fetches team pitching stats (proxy for bullpen strength)."""
        if not team_id:
            return {"team_era": np.nan}
        
        # We use the team stats endpoint
        stats_data = self._get_json(f"teams/{team_id}/stats", {"group": "pitching"})
        
        team_stats = {"team_era": np.nan}
        
        try:
            if stats_data and 'stats' in stats_data and len(stats_data['stats']) > 0:
                stat_group = stats_data['stats'][0]['stat']
                team_stats["team_era"] = float(stat_group.get('era', np.nan))
        except (KeyError, ValueError, IndexError) as e:
            print(f"Warning: Could not parse team stats for team {team_id}: {e}")
            
        return team_stats

    def fetch_today_schedule(self):
        """Main orchestrator to fetch schedule and consolidate data."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"Fetching schedule for: {today_str}")
        
        schedule_data = self._get_json("schedule", {"sportId": 1, "date": today_str})
        
        if not schedule_data or 'dates' not in schedule_data or len(schedule_data['dates']) == 0:
            print("No games found for today.")
            return pd.DataFrame()

        games_list = []
        
        # Iterate through games
        for date_obj in schedule_data['dates']:
            for game in date_obj.get('games', []):
                try:
                    game_id = game.get('gamePk')
                    venue_id = game.get('venue', {}).get('id', np.nan)
                    
                    # Extract Teams
                    away_team = game.get('teams', {}).get('away', {})
                    home_team = game.get('teams', {}).get('home', {})
                    
                    away_id = away_team.get('team', {}).get('id')
                    home_id = home_team.get('team', {}).get('id')
                    
                    away_name = away_team.get('team', {}).get('name')
                    home_name = home_team.get('team', {}).get('name')

                    # Extract Probable Pitchers
                    # Note: Probable pitchers are often in the 'probablePitcher' field
                    away_sp = away_team.get('probablePitcher', {})
                    home_sp = home_team.get('probablePitcher', {})
                    
                    away_sp_id = away_sp.get('id')
                    away_sp_name = away_sp.get('fullName')
                    
                    home_sp_id = home_sp.get('id')
                    home_sp_name = home_sp.get('fullName')

                    # Fetch Pitcher Stats
                    away_sp_stats = self.get_pitcher_stats(away_sp_id)
                    home_sp_stats = self.get_pitcher_stats(home_sp_id)

                    # Fetch Team Bullpen/Pitching Stats
                    away_team_stats = self.get_team_bullpen_stats(away_id)
                    home_team_stats = self.get_team_bullpen_stats(home_id)

                    # Consolidate all data into a single dictionary
                    game_row = {
                        "game_id": game_id,
                        "date": today_str,
                        "venue_id": venue_id,
                        "away_team": away_name,
                        "home_team": home_name,
                        "away_sp_name": away_sp_name,
                        "away_sp_id": away_sp_id,
                        "away_sp_era": away_sp_stats['era'],
                        "away_sp_whip": away_sp_stats['whip'],
                        "away_sp_w": away_sp_stats['wins'],
                        "away_sp_l": away_sp_stats['losses'],
                        "away_team_era": away_team_stats['team_era'],
                        "home_sp_name": home_sp_name,
                        "home_sp_id": home_sp_id,
                        "home_sp_era": home_sp_stats['era'],
                        "home_sp_whip": home_sp_stats['whip'],
                        "home_sp_w": home_sp_stats['wins'],
                        "home_sp_l": home_sp_stats['losses'],
                        "home_team_era": home_team_stats['team_era']
                    }
                    
                    games_list.append(game_row)
                    print(f"Successfully processed: {away_name} @ {home_name}")

                except Exception as e:
                    print(f"Error processing game {game.get('gamePk')}: {e}")
                    continue

        df = pd.DataFrame(games_list)
        
        if not df.empty:
            df.to_csv(self.output_path, index=False)
            print(f"Data successfully saved to {self.output_path}")
        else:
            print("No data collected.")
            
        return df

if __name__ == "__main__":
    fetcher = MLBDataFetcher()
    fetcher.fetch_today_schedule()
