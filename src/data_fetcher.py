import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MLBDataFetcher:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.current_season = datetime.now().year
        self.output_path = "data/today_games.csv"
        os.makedirs("data", exist_ok=True)

    def _get_json(self, endpoint, params=None):
        """Fetch JSON data with retry logic and error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/{endpoint}", 
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts for {endpoint}: {e}")
                    return None
                print(f"Retry {attempt + 1}/{max_retries - 1} for {endpoint}...")
                continue

    def get_pitcher_stats(self, player_id):
        """Extract comprehensive pitcher statistics with robust error handling"""
        if not player_id:
            return {
                "era": np.nan, "whip": np.nan, "wins": 0, "losses": 0,
                "strikeouts": 0, "walks": 0, "innings_pitched": 0,
                "games_started": 0, "quality_starts": 0
            }
        
        stats_data = self._get_json(f"people/{player_id}", {"hydrate": "stats(group=[pitching])"})
        pitcher_stats = {
            "era": np.nan, "whip": np.nan, "wins": 0, "losses": 0,
            "strikeouts": 0, "walks": 0, "innings_pitched": 0,
            "games_started": 0, "quality_starts": 0
        }
        
        try:
            if stats_data and 'stats' in stats_data and len(stats_data['stats']) > 0:
                # Find the pitching stat group
                for stat_group in stats_data['stats']:
                    if stat_group.get('group', {}).get('displayName') == 'Pitching':
                        stat = stat_group.get('stats', [{}])[0].get('stats', {})
                        pitcher_stats.update({
                            "era": float(stat.get('era', np.nan)) if stat.get('era') else np.nan,
                            "whip": float(stat.get('whip', np.nan)) if stat.get('whip') else np.nan,
                            "wins": int(stat.get('wins', 0)),
                            "losses": int(stat.get('losses', 0)),
                            "strikeouts": int(stat.get('strikeOuts', 0)),
                            "walks": int(stat.get('baseOnBalls', 0)),
                            "innings_pitched": float(stat.get('inningsPitched', 0)),
                            "games_started": int(stat.get('gamesStarted', 0)),
                            "quality_starts": int(stat.get('qualityStarts', 0))
                        })
                        break
        except Exception as e:
            print(f"Warning: Could not parse stats for player {player_id}: {e}")
            
        return pitcher_stats

    def get_team_stats(self, team_id):
        """Extract team batting and pitching statistics"""
        if not team_id:
            return {
                "team_era": np.nan, "team_whip": np.nan, "team_runs": 0,
                "team_avg": np.nan, "team_obp": np.nan, "team_slg": np.nan
            }
        
        team_stats = {
            "team_era": np.nan, "team_whip": np.nan, "team_runs": 0,
            "team_avg": np.nan, "team_obp": np.nan, "team_slg": np.nan
        }
        
        try:
            # Get team season stats
            stats_data = self._get_json(f"teams/{team_id}/stats", {
                "group": "pitching", 
                "season": self.current_season
            })
            
            if stats_data and 'stats' in stats_data:
                for stat_group in stats_data['stats']:
                    if stat_group.get('type', {}).get('displayName') == 'Season':
                        stat = stat_group.get('stats', {})
                        team_stats["team_era"] = float(stat.get('era', np.nan)) if stat.get('era') else np.nan
                        team_stats["team_whip"] = float(stat.get('whip', np.nan)) if stat.get('whip') else np.nan
                        break
            
            # Get team batting stats
            batting_data = self._get_json(f"teams/{team_id}/stats", {
                "group": "batting",
                "season": self.current_season
            })
            
            if batting_data and 'stats' in batting_data:
                for stat_group in batting_data['stats']:
                    if stat_group.get('type', {}).get('displayName') == 'Season':
                        stat = stat_group.get('stats', {})
                        team_stats["team_avg"] = float(stat.get('avg', np.nan)) if stat.get('avg') else np.nan
                        team_stats["team_obp"] = float(stat.get('obp', np.nan)) if stat.get('obp') else np.nan
                        team_stats["team_slg"] = float(stat.get('slg', np.nan)) if stat.get('slg') else np.nan
                        break
        except Exception as e:
            print(f"Warning: Could not parse team stats for team {team_id}: {e}")
            
        return team_stats

    def get_recent_performance(self, team_id, days=14):
        """Get recent team performance (last N days)"""
        if not team_id:
            return {"recent_runs_per_game": np.nan, "recent_era": np.nan}
        
        recent_stats = {"recent_runs_per_game": np.nan, "recent_era": np.nan}
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            schedule_data = self._get_json("schedule", {
                "sportId": 1,
                "teamId": team_id,
                "startDate": start_date.strftime('%Y-%m-%d'),
                "endDate": end_date.strftime('%Y-%m-%d')
            })
            
            if schedule_data and 'dates' in schedule_data:
                runs_scored = 0
                games_played = 0
                
                for date_obj in schedule_data['dates']:
                    for game in date_obj.get('games', []):
                        if game.get('status', {}).get('abstractGameState') == 'Final':
                            games_played += 1
                            teams_info = game.get('teams', {})
                            # Check if this is home or away team
                            if teams_info.get('home', {}).get('team', {}).get('id') == team_id:
                                runs_scored += teams_info.get('home', {}).get('runs', 0)
                            else:
                                runs_scored += teams_info.get('away', {}).get('runs', 0)
                
                if games_played > 0:
                    recent_stats["recent_runs_per_game"] = runs_scored / games_played
        except Exception as e:
            print(f"Warning: Could not get recent performance for team {team_id}: {e}")
            
        return recent_stats

    def fetch_today_schedule(self):
        """Fetch today's MLB schedule with comprehensive statistics"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"Fetching schedule for: {today_str}")
        
        schedule_data = self._get_json("schedule", {"sportId": 1, "date": today_str})
        
        if not schedule_data or 'dates' not in schedule_data or len(schedule_data['dates']) == 0:
            print("No games found for today.")
            return pd.DataFrame()

        games_list = []
        for date_obj in schedule_data['dates']:
            for game in date_obj.get('games', []):
                try:
                    game_id = game.get('gamePk')
                    venue_id = game.get('venue', {}).get('id', np.nan)
                    away_team = game.get('teams', {}).get('away', {})
                    home_team = game.get('teams', {}).get('home', {})
                    
                    away_id = away_team.get('team', {}).get('id')
                    home_id = home_team.get('team', {}).get('id')
                    away_name = away_team.get('team', {}).get('name')
                    home_name = home_team.get('team', {}).get('name')

                    away_sp = away_team.get('probablePitcher', {})
                    home_sp = home_team.get('probablePitcher', {})
                    away_sp_id = away_sp.get('id')
                    away_sp_name = away_sp.get('fullName', 'TBD')
                    home_sp_id = home_sp.get('id')
                    home_sp_name = home_sp.get('fullName', 'TBD')

                    # Fetch comprehensive statistics
                    away_sp_stats = self.get_pitcher_stats(away_sp_id)
                    home_sp_stats = self.get_pitcher_stats(home_sp_id)
                    away_team_stats = self.get_team_stats(away_id)
                    home_team_stats = self.get_team_stats(home_id)
                    away_recent = self.get_recent_performance(away_id)
                    home_recent = self.get_recent_performance(home_id)

                    game_row = {
                        # Basic game info
                        "game_id": game_id,
                        "date": today_str,
                        "venue_id": venue_id,
                        "away_team": away_name,
                        "home_team": home_name,
                        # Away pitcher stats
                        "away_sp_name": away_sp_name,
                        "away_sp_id": away_sp_id,
                        "away_sp_era": away_sp_stats['era'],
                        "away_sp_whip": away_sp_stats['whip'],
                        "away_sp_w": away_sp_stats['wins'],
                        "away_sp_l": away_sp_stats['losses'],
                        "away_sp_k9": away_sp_stats['strikeouts'] / away_sp_stats['innings_pitched'] * 9 if away_sp_stats['innings_pitched'] > 0 else np.nan,
                        "away_sp_bb9": away_sp_stats['walks'] / away_sp_stats['innings_pitched'] * 9 if away_sp_stats['innings_pitched'] > 0 else np.nan,
                        "away_sp_qs_pct": away_sp_stats['quality_starts'] / away_sp_stats['games_started'] if away_sp_stats['games_started'] > 0 else np.nan,
                        # Away team stats
                        "away_team_era": away_team_stats['team_era'],
                        "away_team_whip": away_team_stats['team_whip'],
                        "away_team_avg": away_team_stats['team_avg'],
                        "away_team_obp": away_team_stats['team_obp'],
                        "away_team_slg": away_team_stats['team_slg'],
                        "away_recent_rpg": away_recent['recent_runs_per_game'],
                        # Home pitcher stats
                        "home_sp_name": home_sp_name,
                        "home_sp_id": home_sp_id,
                        "home_sp_era": home_sp_stats['era'],
                        "home_sp_whip": home_sp_stats['whip'],
                        "home_sp_w": home_sp_stats['wins'],
                        "home_sp_l": home_sp_stats['losses'],
                        "home_sp_k9": home_sp_stats['strikeouts'] / home_sp_stats['innings_pitched'] * 9 if home_sp_stats['innings_pitched'] > 0 else np.nan,
                        "home_sp_bb9": home_sp_stats['walks'] / home_sp_stats['innings_pitched'] * 9 if home_sp_stats['innings_pitched'] > 0 else np.nan,
                        "home_sp_qs_pct": home_sp_stats['quality_starts'] / home_sp_stats['games_started'] if home_sp_stats['games_started'] > 0 else np.nan,
                        # Home team stats
                        "home_team_era": home_team_stats['team_era'],
                        "home_team_whip": home_team_stats['team_whip'],
                        "home_team_avg": home_team_stats['team_avg'],
                        "home_team_obp": home_team_stats['team_obp'],
                        "home_team_slg": home_team_stats['team_slg'],
                        "home_recent_rpg": home_recent['recent_runs_per_game'],
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
        return df

if __name__ == "__main__":
    fetcher = MLBDataFetcher()
    fetcher.fetch_today_schedule()
