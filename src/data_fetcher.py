import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple

class MLBDataFetcher:
    """Advanced MLB data fetcher with intelligent pitcher resolution"""
    
    # Cache for team roster (reduces API calls)
    _roster_cache = {}
    _last_roster_fetch = {}
    ROSTER_CACHE_HOURS = 6
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.current_season = datetime.now().year
        self.output_path = "data/today_games.csv"
        os.makedirs("data", exist_ok=True)
        self.processed_game_ids = set()  # Track processed games to avoid duplicates

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
                    print(f"❌ Failed after {max_retries} attempts for {endpoint}: {e}")
                    return None
                print(f"🔄 Retry {attempt + 1}/{max_retries - 1} for {endpoint}...")
                continue

    def _get_team_roster(self, team_id: int, use_cache=True) -> Optional[Dict]:
        """
        Fetch team roster with caching to minimize API calls.
        Returns roster hydrated with pitching stats.
        """
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
            print(f"⚠️  Warning: Could not fetch roster for team {team_id}: {e}")
        
        return None

    def _get_recent_pitcher_ids(self, team_id: int, days: int = 14) -> List[int]:
        """
        Get pitcher IDs who have recently pitched for the team.
        Returns list of pitcher IDs sorted by recent appearances.
        """
        recent_pitchers = []
        
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
                pitcher_count = {}
                
                for date_obj in schedule_data['dates']:
                    for game in date_obj.get('games', []):
                        # Get boxscore for detailed pitcher info
                        game_id = game.get('gamePk')
                        boxscore = self._get_json(f"game/{game_id}/boxscore")
                        
                        if boxscore:
                            teams_box = boxscore.get('teams', {})
                            # Check if our team was home or away
                            for team_side in ['home', 'away']:
                                team_info = teams_box.get(team_side, {}).get('team', {})
                                if team_info.get('id') == team_id:
                                    # Get all pitchers from this game
                                    players = boxscore.get('teams', {}).get(team_side, {}).get('players', {})
                                    for player_key, player_info in players.items():
                                        if player_key.startswith('ID'):
                                            pitcher_stats = player_info.get('stats', {}).get('pitching')
                                            if pitcher_stats:  # Is a pitcher
                                                player_id = player_info.get('person', {}).get('id')
                                                if player_id:
                                                    pitcher_count[player_id] = pitcher_count.get(player_id, 0) + 1
                
                # Sort by recent appearances
                recent_pitchers = sorted(pitcher_count.items(), key=lambda x: x[1], reverse=True)
                recent_pitchers = [pid for pid, count in recent_pitchers[:10]]  # Top 10 recent
        
        except Exception as e:
            print(f"⚠️  Warning: Could not get recent pitchers for team {team_id}: {e}")
        
        return recent_pitchers

    def _get_team_starter_by_games_started(self, team_id: int) -> Optional[Tuple[int, str, Dict]]:
        """
        Find the pitcher with most games started this season.
        Returns: (pitcher_id, pitcher_name, stats_dict)
        """
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
                
                # Get pitching stats
                stats_obj = person.get('stats', [])
                for stat_group in stats_obj:
                    if stat_group.get('type', {}).get('displayName') == 'season':
                        stat = stat_group.get('stats', [{}])[0] if stat_group.get('stats') else {}
                        games_started = stat.get('gamesStarted', 0)
                        
                        # Only consider pitchers with at least 2 starts
                        if games_started > best_starts:
                            best_starts = games_started
                            best_pitcher = (pitcher_id, pitcher_name, stat)
            
            if best_pitcher and best_pitcher[2]:
                return best_pitcher
        
        except Exception as e:
            print(f"⚠️  Warning: Could not find starter for team {team_id}: {e}")
        
        return None

    def _get_bullpen_average_stats(self, team_id: int, top_n: int = 3) -> Dict:
        """
        Calculate weighted average stats of top relief pitchers.
        Returns weighted ERA and SIERA for Bullpen Days.
        """
        bullpen_stats = {
            "era": np.nan, "whip": np.nan, "siera": np.nan,
            "strikeouts": 0, "walks": 0, "innings_pitched": 0,
            "appearances": 0, "source": "bullpen_avg"
        }
        
        try:
            recent_pitchers = self._get_recent_pitcher_ids(team_id, days=7)
            
            if not recent_pitchers:
                return bullpen_stats
            
            # Fetch stats for recent pitchers and find relief specialists
            relievers = []
            
            for pitcher_id in recent_pitchers[:15]:  # Check top 15 recent
                stats = self.get_pitcher_stats(pitcher_id)
                
                # Identify relievers (low games started, multiple appearances)
                if stats.get('games_started', 0) <= 1 and stats.get('appearances', 0) >= 5:
                    relievers.append(stats)
            
            if not relievers:
                return bullpen_stats
            
            # Use top N relievers by ERA (best performance)
            relievers_sorted = sorted(
                relievers, 
                key=lambda x: x.get('era', float('inf'))
            )[:top_n]
            
            if not relievers_sorted:
                return bullpen_stats
            
            # Calculate weighted average
            total_weight = 0
            weighted_era = 0
            weighted_whip = 0
            total_k = 0
            total_bb = 0
            total_ip = 0
            total_apps = 0
            
            for reliever in relievers_sorted:
                # Weight by innings pitched
                weight = reliever.get('innings_pitched', 0)
                if weight > 0:
                    total_weight += weight
                    weighted_era += reliever.get('era', 0) * weight
                    weighted_whip += reliever.get('whip', 1.30) * weight
                    total_k += reliever.get('strikeouts', 0)
                    total_bb += reliever.get('walks', 0)
                    total_ip += reliever.get('innings_pitched', 0)
                    total_apps += reliever.get('appearances', 0)
            
            if total_weight > 0:
                bullpen_stats["era"] = weighted_era / total_weight
                bullpen_stats["whip"] = weighted_whip / total_weight
                bullpen_stats["strikeouts"] = int(total_k)
                bullpen_stats["walks"] = int(total_bb)
                bullpen_stats["innings_pitched"] = total_ip
                bullpen_stats["appearances"] = total_apps
                
                # Calculate simple SIERA proxy: (13*K - 3*BB) / IP, if available
                if total_ip > 0:
                    bullpen_stats["siera"] = (13 * total_k - 3 * total_bb) / total_ip
        
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate bullpen average for team {team_id}: {e}")
        
        return bullpen_stats

    def get_pitcher_stats(self, player_id: Optional[int]) -> Dict:
        """Extract comprehensive pitcher statistics with robust error handling"""
        if not player_id:
            return {
                "era": np.nan, "whip": np.nan, "wins": 0, "losses": 0,
                "strikeouts": 0, "walks": 0, "innings_pitched": 0,
                "games_started": 0, "appearances": 0, "quality_starts": 0
            }
        
        stats_data = self._get_json(f"people/{player_id}", {
            "hydrate": "stats(group=[pitching],type=[season])"
        })
        
        pitcher_stats = {
            "era": np.nan, "whip": np.nan, "wins": 0, "losses": 0,
            "strikeouts": 0, "walks": 0, "innings_pitched": 0,
            "games_started": 0, "appearances": 0, "quality_starts": 0
        }
        
        try:
            if stats_data and 'stats' in stats_data and len(stats_data['stats']) > 0:
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
                            "appearances": int(stat.get('gameType', 0)) or int(stat.get('appearances', 0)),
                            "quality_starts": int(stat.get('qualityStarts', 0))
                        })
                        break
        except Exception as e:
            print(f"⚠️  Warning: Could not parse stats for player {player_id}: {e}")
            
        return pitcher_stats

    def _resolve_pitcher(self, team_id: int, probable_pitcher_id: Optional[int], 
                        probable_pitcher_name: Optional[str]) -> Tuple[Optional[int], str, Dict, str]:
        """
        Intelligently resolve pitcher with fallback strategy:
        1. Use probable starter if valid
        2. Search for team's recent starter by games started
        3. Fall back to bullpen average if no starter found
        
        Returns: (pitcher_id, pitcher_name, stats_dict, resolution_method)
        """
        method = "probable"
        
        # Step 1: Try to use probable pitcher if provided
        if probable_pitcher_id:
            stats = self.get_pitcher_stats(probable_pitcher_id)
            if not np.isnan(stats.get('era', np.nan)) and stats.get('games_started', 0) > 0:
                return probable_pitcher_id, probable_pitcher_name or "Unknown", stats, method
        
        method = "games_started"
        # Step 2: Find team's leading starter by games started this season
        starter_result = self._get_team_starter_by_games_started(team_id)
        if starter_result:
            pitcher_id, pitcher_name, stat_dict = starter_result
            stats = self.get_pitcher_stats(pitcher_id)
            if stats and not np.isnan(stats.get('era', np.nan)):
                return pitcher_id, pitcher_name, stats, method
        
        method = "bullpen_day"
        # Step 3: No valid starter found - calculate bullpen average
        bullpen_stats = self._get_bullpen_average_stats(team_id)
        bullpen_stats['games_started'] = 0  # Mark as bullpen
        bullpen_stats['appearances'] = bullpen_stats.get('appearances', 0)
        
        return None, "Bullpen Day", bullpen_stats, method

    def get_team_stats(self, team_id: Optional[int]) -> Dict:
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
            print(f"⚠️  Warning: Could not parse team stats for team {team_id}: {e}")
            
        return team_stats

    def get_recent_performance(self, team_id: Optional[int], days: int = 14) -> Dict:
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
            print(f"⚠️  Warning: Could not get recent performance for team {team_id}: {e}")
            
        return recent_stats

    def fetch_today_schedule(self) -> pd.DataFrame:
        """Fetch today's MLB schedule with comprehensive statistics and intelligent pitcher resolution"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"\n📅 Fetching schedule for: {today_str}\n")
        
        schedule_data = self._get_json("schedule", {"sportId": 1, "date": today_str})
        
        if not schedule_data or 'dates' not in schedule_data or len(schedule_data['dates']) == 0:
            print("⚠️  No games found for today.")
            return pd.DataFrame()

        games_list = []
        game_count = 0
        
        for date_obj in schedule_data['dates']:
            for game in date_obj.get('games', []):
                try:
                    game_id = game.get('gamePk')
                    
                    # Avoid duplicate processing
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

                    # Get probable pitchers from API
                    away_sp = away_team.get('probablePitcher', {})
                    home_sp = home_team.get('probablePitcher', {})
                    
                    away_sp_id = away_sp.get('id')
                    away_sp_name = away_sp.get('fullName')
                    home_sp_id = home_sp.get('id')
                    home_sp_name = home_sp.get('fullName')

                    # 🎯 INTELLIGENT PITCHER RESOLUTION
                    away_sp_id_resolved, away_sp_name_resolved, away_sp_stats, away_method = \
                        self._resolve_pitcher(away_id, away_sp_id, away_sp_name)
                    
                    home_sp_id_resolved, home_sp_name_resolved, home_sp_stats, home_method = \
                        self._resolve_pitcher(home_id, home_sp_id, home_sp_name)

                    # Fetch team stats
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
                        
                        # Away pitcher stats with resolution method
                        "away_sp_name": away_sp_name_resolved,
                        "away_sp_id": away_sp_id_resolved,
                        "away_sp_era": away_sp_stats['era'],
                        "away_sp_whip": away_sp_stats['whip'],
                        "away_sp_w": away_sp_stats.get('wins', 0),
                        "away_sp_l": away_sp_stats.get('losses', 0),
                        "away_sp_k9": away_sp_stats.get('strikeouts', 0) / away_sp_stats.get('innings_pitched', 1) * 9 if away_sp_stats.get('innings_pitched', 0) > 0 else np.nan,
                        "away_sp_bb9": away_sp_stats.get('walks', 0) / away_sp_stats.get('innings_pitched', 1) * 9 if away_sp_stats.get('innings_pitched', 0) > 0 else np.nan,
                        "away_sp_qs_pct": away_sp_stats.get('quality_starts', 0) / max(away_sp_stats.get('games_started', 1), 1) if away_sp_stats.get('games_started', 0) > 0 else np.nan,
                        "away_pitcher_type": away_method,  # NEW: Track resolution method
                        
                        # Away team stats
                        "away_team_era": away_team_stats['team_era'],
                        "away_team_whip": away_team_stats['team_whip'],
                        "away_team_avg": away_team_stats['team_avg'],
                        "away_team_obp": away_team_stats['team_obp'],
                        "away_team_slg": away_team_stats['team_slg'],
                        "away_recent_rpg": away_recent['recent_runs_per_game'],
                        
                        # Home pitcher stats with resolution method
                        "home_sp_name": home_sp_name_resolved,
                        "home_sp_id": home_sp_id_resolved,
                        "home_sp_era": home_sp_stats['era'],
                        "home_sp_whip": home_sp_stats['whip'],
                        "home_sp_w": home_sp_stats.get('wins', 0),
                        "home_sp_l": home_sp_stats.get('losses', 0),
                        "home_sp_k9": home_sp_stats.get('strikeouts', 0) / home_sp_stats.get('innings_pitched', 1) * 9 if home_sp_stats.get('innings_pitched', 0) > 0 else np.nan,
                        "home_sp_bb9": home_sp_stats.get('walks', 0) / home_sp_stats.get('innings_pitched', 1) * 9 if home_sp_stats.get('innings_pitched', 0) > 0 else np.nan,
                        "home_sp_qs_pct": home_sp_stats.get('quality_starts', 0) / max(home_sp_stats.get('games_started', 1), 1) if home_sp_stats.get('games_started', 0) > 0 else np.nan,
                        "home_pitcher_type": home_method,  # NEW: Track resolution method
                        
                        # Home team stats
                        "home_team_era": home_team_stats['team_era'],
                        "home_team_whip": home_team_stats['team_whip'],
                        "home_team_avg": home_team_stats['team_avg'],
                        "home_team_obp": home_team_stats['team_obp'],
                        "home_team_slg": home_team_stats['team_slg'],
                        "home_recent_rpg": home_recent['recent_runs_per_game'],
                    }
                    
                    games_list.append(game_row)
                    
                    # Logging with resolution methods
                    print(f"✅ Game {game_count}: {away_name} ({away_method}) @ {home_name} ({home_method})")
                    print(f"   🎯 Pitchers: {away_sp_name_resolved} vs {home_sp_name_resolved}")
                    
                except Exception as e:
                    print(f"❌ Error processing game {game.get('gamePk')}: {e}")
                    import traceback
                    traceback.print_exc()
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
