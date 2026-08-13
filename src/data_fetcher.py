import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class MLBDataFetcher:
    """Advanced MLB data fetcher with intelligent pitcher resolution"""
    
    _roster_cache = {}
    _last_roster_fetch = {}
    ROSTER_CACHE_HOURS = 6
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.current_season = datetime.now().year
        self.output_path = "data/today_games.csv"
        os.makedirs("data", exist_ok=True)
        self.processed_game_ids = set()

    def _get_json(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch JSON data with retry logic and clean endpoint parsing"""
        max_retries = 3
        endpoint_clean = endpoint.lstrip('/')
        url = f"{self.base_url}/{endpoint_clean}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed after {max_retries} attempts for {endpoint_clean}: {e}")
                    return None
                continue

    def get_pitcher_stats(self, player_id: Optional[int]) -> Dict:
        """Extract comprehensive pitcher statistics from person hydrate endpoint"""
        pitcher_stats = {
            "era": np.nan, "whip": np.nan, "wins": 0, "losses": 0,
            "strikeouts": 0, "walks": 0, "innings_pitched": 0.0,
            "games_started": 0, "appearances": 0, "quality_starts": 0
        }
        
        if not player_id:
            return pitcher_stats
        
        stats_data = self._get_json(f"people/{player_id}", {
            "hydrate": "stats(group=[pitching],type=[season])"
        })
        
        try:
            if stats_data and 'people' in stats_data and len(stats_data['people']) > 0:
                person = stats_data['people'][0]
                stats_list = person.get('stats', [])
                for stat_group in stats_list:
                    if stat_group.get('group', {}).get('displayName') == 'pitching':
                        splits = stat_group.get('splits', [])
                        if splits:
                            stat = splits[0].get('stat', {})
                            pitcher_stats.update({
                                "era": float(stat.get('era', np.nan)) if stat.get('era') is not None else np.nan,
                                "whip": float(stat.get('whip', np.nan)) if stat.get('whip') is not None else np.nan,
                                "wins": int(stat.get('wins', 0)),
                                "losses": int(stat.get('losses', 0)),
                                "strikeouts": int(stat.get('strikeOuts', 0)),
                                "walks": int(stat.get('baseOnBalls', 0)),
                                "innings_pitched": float(stat.get('inningsPitched', 0.0)),
                                "games_started": int(stat.get('gamesStarted', 0)),
                                "appearances": int(stat.get('gamesPitched', 0)),
                                "quality_starts": int(stat.get('qualityStarts', 0))
                            })
                            break
        except Exception as e:
            print(f"⚠️ Warning: Could not parse stats for player {player_id}: {e}")
            
        return pitcher_stats

    def _get_team_roster(self, team_id: int, use_cache=True) -> Optional[Dict]:
        """Fetch team roster with caching to minimize API calls"""
        if use_cache and team_id in self._roster_cache:
            cache_time = self._last_roster_fetch.get(team_id)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.ROSTER_CACHE_HOURS * 3600:
                return self._roster_cache[team_id]
        
        try:
            roster_data = self._get_json(f"teams/{team_id}/roster", {
                "hydrate": "person(stats(group=[pitching],type=[season]))"
            })
            if roster_data and 'roster' in roster_data:
                self._roster_cache[team_id] = roster_data
                self._last_roster_fetch[team_id] = datetime.now()
                return roster_data
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch roster for team {team_id}: {e}")
        
        return None

    def _get_team_starter_by_games_started(self, team_id: int) -> Optional[Tuple[int, str, Dict]]:
        """Find the pitcher with most games started on the current roster"""
        try:
            roster_data = self._get_team_roster(team_id)
            if not roster_data:
                return None
            
            best_pitcher = None
            best_starts = -1
            
            for player in roster_data.get('roster', []):
                person = player.get('person', {})
                pitcher_id = person.get('id')
                pitcher_name = person.get('fullName', 'Unknown')
                
                stats_obj = person.get('stats', [])
                for stat_group in stats_obj:
                    if stat_group.get('group', {}).get('displayName') == 'pitching':
                        splits = stat_group.get('splits', [])
                        if splits:
                            stat = splits[0].get('stat', {})
                            games_started = stat.get('gamesStarted', 0)
                            if games_started > best_starts:
                                best_starts = games_started
                                best_pitcher = (pitcher_id, pitcher_name, stat)
            
            if best_pitcher and best_starts > 0:
                return best_pitcher
        except Exception as e:
            print(f"⚠️ Warning: Could not find starter for team {team_id}: {e}") me
        
        return None

    def _get_bullpen_average_stats(self, team_id: int) -> Dict:
        """Fallback stats for Bullpen Days without API hammering"""
        return {
            "era": 4.15,
            "whip": 1.28,
            "wins": 0,
            "losses": 0,
            "strikeouts": 8,
            "walks": 3,
            "innings_pitched": 9.0,
            "games_started": 0,
            "appearances": 1,
            "quality_starts": 0
        }

    def _resolve_pitcher(self, team_id: int, probable_pitcher_id: Optional[int], 
                        probable_pitcher_name: Optional[str]) -> Tuple[Optional[int], str, Dict, str]:
        """
        Intelligently resolve pitcher:
        1. Direct probable pitcher if provided by MLB
        2. Leading starter by Games Started
        3. Bullpen Day fallback with league-average metrics
        """
        method = "probable"
        
        if probable_pitcher_id and probable_pitcher_name:
            stats = self.get_pitcher_stats(probable_pitcher_id)
            if np.isnan(stats.get('era', np.nan)):
                stats['era'] = 4.20
                stats['whip'] = 1.28
            return probable_pitcher_id, probable_pitcher_name, stats, method
        
        method = "games_started"
        starter_result = self._get_team_starter_by_games_started(team_id)
        if starter_result:
            pitcher_id, pitcher_name, stat_dict = starter_result
            stats = self.get_pitcher_stats(pitcher_id)
            if stats and not np.isnan(stats.get('era', np.nan)):
                return pitcher_id, pitcher_name, stats, method
        
        method = "bullpen_day"
        bullpen_stats = self._get_bullpen_average_stats(team_id)
        return None, "Bullpen Day", bullpen_stats, method

    def get_team_stats(self, team_id: Optional[int]) -> Dict:
        """Extract team batting and pitching statistics"""
        team_stats = {
            "team_era": 4.10, "team_whip": 1.25,
            "team_avg": .248, "team_obp": .318, "team_slg": .410
        }
        if not team_id:
            return team_stats
        
        try:
            pitching_data = self._get_json(f"teams/{team_id}/stats", {
                "stats": "season",
                "group": "pitching", 
                "season": self.current_season
            })
            
            if pitching_data and 'stats' in pitching_data:
                for stat_group in pitching_data['stats']:
                    splits = stat_group.get('splits', [])
                    if splits:
                        stat = splits[0].get('stat', {})
                        team_stats["team_era"] = float(stat.get('era', 4.10))
                        team_stats["team_whip"] = float(stat.get('whip', 1.25))
                        break
            
            batting_data = self._get_json(f"teams/{team_id}/stats", {
                "stats": "season",
                "group": "batting",
                "season": self.current_season
            })
            
            if batting_data and 'stats' in batting_data:
                for stat_group in batting_data['stats']:
                    splits = stat_group.get('splits', [])
                    if splits:
                        stat = splits[0].get('stat', {})
                        team_stats["team_avg"] = float(stat.get('avg', .248))
                        team_stats["team_obp"] = float(stat.get('obp', .318))
                        team_stats["team_slg"] = float(stat.get('slg', .410))
                        break
        except Exception as e:
            print(f"⚠️ Warning: Could not parse team stats for team {team_id}: {e}")
            
        return team_stats

    def get_recent_performance(self, team_id: Optional[int], days: int = 14) -> Dict:
        """Get recent team performance (last N days)"""
        recent_stats = {"recent_runs_per_game": 4.5}
        if not team_id:
            return recent_stats
        
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
                            if teams_info.get('home', {}).get('team', {}).get('id') == team_id:
                                runs_scored += teams_info.get('home', {}).get('score', 0)
                            else:
                                runs_scored += teams_info.get('away', {}).get('score', 0)
                
                if games_played > 0:
                    recent_stats["recent_runs_per_game"] = round(runs_scored / games_played, 2)
        except Exception as e:
            print(f"⚠️ Warning: Could not get recent performance for team {team_id}: {e}")
            
        return recent_stats

    def fetch_today_schedule(() -> pd.DataFrame:
        """Fetch today's MLB schedule with comprehensive statistics"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"\n📅 Fetching schedule for: {today_str}\n")
        
        schedule_data = self._get_json("schedule", {
            "sportId": 1, 
            "date": today_str,
            "hydrate": "probablePitcher"
        })
        
        if not schedule_data or 'dates' not in schedule_data or len(schedule_data['dates']) == 0:
            print("⚠️ No games found for today.")
            return pd.DataFrame()

        games_list = []
        game_count = 0
        
        for date_obj in schedule_data['dates']:
            for game in date_obj.get('games', []):
                try:
                    game_id = game.get('gamePk')
                    
                    if game_id in self.processed_game_ids:
                        continue
                    self.processed_game_ids.add(game_id)
                    
                    game_count += 1
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
                    away_sp_name = away_sp.get('fullName')
                    home_sp_id = home_sp.get('id')
                    home_sp_name = home_sp.get('fullName')

                    away_sp_id_res, away_sp_name_res, away_sp_stats, away_method = \
                        self._resolve_pitcher(away_id, away_sp_id, away_sp_name)
                    
                    home_sp_id_res, home_sp_name_res, home_sp_stats, home_method = \
                        self._resolve_pitcher(home_id, home_sp_id, home_sp_name)

                    away_team_stats = self.get_team_stats(away_id)
                    home_team_stats = self.get_team_stats(home_id)
                    away_recent = self.get_recent_performance(away_id)
                    home_recent = self.get_recent_performance(home_id)

                    away_ip = max(away_sp_stats.get('innings_pitched', 1.0), 1.0)
                    home_ip = max(home_sp_stats.get('innings_pitched', 1.0), 1.0)

                    game_row = {
                        "game_id": game_id,
                        "date": today_str,
                        "venue_id": venue_id,
                        "away_team": away_name,
                        "home_team": home_name,
                        
                        "away_sp_name": away_sp_name_res,
                        "away_sp_id": away_sp_id_res,
                        "away_sp_era": away_sp_stats['era'],
                        "away_sp_whip": away_sp_stats['whip'],
                        "away_sp_w": away_sp_stats.get('wins', 0),
                        "away_sp_l": away_sp_stats.get('losses', 0),
                        "away_sp_k9": round((away_sp_stats.get('strikeouts', 0) / away_ip) * 9, 2),
                        "away_sp_bb9": round((away_sp_stats.get('walks', 0) / away_ip) * 9, 2),
                        "away_sp_qs_pct": round(away_sp_stats.get('quality_starts', 0) / max(away_sp_stats.get('games_started', 1), 1), 2),
                        "away_pitcher_type": away_method,
                        
                        "away_team_era": away_team_stats['team_era'],
                        "away_team_whip": away_team_stats['team_whip'],
                        "away_team_avg": away_team_stats['team_avg'],
                        "away_team_obp": away_team_stats['team_obp'],
                        "away_team_slg": away_team_stats['team_slg'],
                        "away_recent_rpg": away_recent['recent_runs_per_game'],
                        
                        "home_sp_name": home_sp_name_res,
                        "home_sp_id": home_sp_id_res,
                        "home_sp_era": home_sp_stats['era'],
                        "home_sp_whip": home_sp_stats['whip'],
                        "home_sp_w": home_sp_stats.get('wins', 0),
                        "home_sp_l": home_sp_stats.get('losses', 0),
                        "home_sp_k9": round((home_sp_stats.get('strikeouts', 0) / home_ip) * 9, 2),
                        "home_sp_bb9": round((home_sp_stats.get('walks', 0) / home_ip) * 9, 2),
                        "home_sp_qs_pct": round(home_sp_stats.get('quality_starts', 0) / max(home_sp_stats.get('games_started', 1), 1), 2),
                        "home_pitcher_type": home_method,
                        
                        "home_team_era": home_team_stats['team_era'],
                        "home_team_whip": home_team_stats['team_whip'],
                        "home_team_avg": home_team_stats['team_avg'],
                        "home_team_obp": home_team_stats['team_obp'],
                        "home_team_slg": home_team_stats['team_slg'],
                        "home_recent_rpg": home_recent['recent_runs_per_game'],
                    }
                    
                    games_list.append(game_row)
                    print(f"✅ Game {game_count}: {away_name} ({away_method}) @ {home_name} ({home_method})")
                    print(f"   🎯 Pitchers: {away_sp_name_res} vs {home_sp_name_res}")
                    
                except Exception as e:
                    print(f"❌ Error processing game {game.get('gamePk')}: {e}")
                    continue

        df = pd.DataFrame(games_list)
        if not df.empty:
            df.to_csv(self.output_path, index=False)
            print(f"\n✅ Successfully processed {len(df)} games")
            print(f"💾 Data saved to {self.output_path}\n")
        return df

if __name__ == "__main__":
    fetcher = MLBDataFetcher()
    fetcher.fetch_today_schedule()
