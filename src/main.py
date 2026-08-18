import os
import sys
import requests
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import MLBDataFetcher
from predictor import MLBProjector

class MLBOrchestrator:
    """Orchestrate data fetching and projection pipeline"""
    
    def __init__(self):
        self.data_dir = "data"
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def send_telegram_message(self, message: str) -> bool:
        """Send formatted message via Telegram bot with robust chunking and safety fallback"""
        if not self.telegram_token or not self.telegram_chat_id:
            print("⚠️ Warning: Telegram credentials not configured.")
            return False
            
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        max_length = 4000
        
        try:
            for i in range(0, len(message), max_length):
                chunk = message[i:i+max_length]
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown"
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 400:
                    # Fallback without Markdown if markdown parsing fails
                    payload.pop("parse_mode")
                    response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
            
            print("✅ Telegram notification sent successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False

    def format_projections_message(self, df: pd.DataFrame) -> str:
        """Format projections into human-readable Telegram message"""
        if df is None or df.empty:
            return "⚾ *MLB Update*\nNo games scheduled for today. 🧢"

        message = "⚾ *MLB DAILY PROJECTIONS* ⚾\n"
        message += f"📅 {datetime.now().strftime('%B %d, %Y')}\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for idx, row in df.iterrows():
            away_team = row.get('away_team', 'TBD')
            home_team = row.get('home_team', 'TBD')
            message += f"🏟️ *Game {idx + 1}: {away_team} @ {home_team}*\n"
            
            away_sp = row.get('away_sp_name', row.get('away_sp', 'TBD'))
            home_sp = row.get('home_sp_name', row.get('home_sp', 'TBD'))
            
            away_era = row.get('away_sp_era', 'N/A')
            home_era = row.get('home_sp_era', 'N/A')
            
            away_rating_str = f"ERA: {away_era:.2f}" if isinstance(away_era, (int, float)) and not pd.isna(away_era) else "N/A"
            home_rating_str = f"ERA: {home_era:.2f}" if isinstance(home_era, (int, float)) and not pd.isna(home_era) else "N/A"
            
            message += f"🥎 *Pitchers:* {away_sp} ({away_rating_str}) vs {home_sp} ({home_rating_str})\n"
            
            away_off = row.get('away_offense_rating', 100.0)
            away_def = row.get('away_defense_rating', 100.0)
            home_off = row.get('home_offense_rating', 100.0)
            home_def = row.get('home_defense_rating', 100.0)
            
            message += f"⚔️ *Ratings:* OFF/DEF\n"
            message += f"   • {away_team}: {away_off:.1f}/{away_def:.1f}\n"
            message += f"   • {home_team}: {home_off:.1f}/{home_def:.1f}\n"
            
            away_prob = round(float(row.get('prob_away_win', 0.50)) * 100)
            home_prob = round(float(row.get('prob_home_win', 0.50)) * 100)
            message += f"🎯 *Win Probability:*\n"
            message += f"   • {away_team}: {away_prob}%\n"
            message += f"   • {home_team}: {home_prob}%\n"
            
            total = row.get('projected_total', 'N/A')
            if isinstance(total, (int, float)) and not pd.isna(total):
                total_str = f"{total:.1f}"
            else:
                total_str = str(total)
                
            confidence = row.get('confidence', 'N/A')
            if isinstance(confidence, (int, float)) and not pd.isna(confidence):
                conf_str = f"{confidence:.2f}"
            else:
                conf_str = str(confidence)

            message += f"📊 *Total Runs:* {total_str} (Confidence: {conf_str})\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        message += "🔬 *Advanced projection system using:*\n"
        message += "• Pitcher efficiency metrics (ERA, WHIP, K9, BB9)\n"
        message += "• Team offensive/defensive ratings\n"
        message += "• Recent performance trends\n"
        message += "• Stadium park factors\n"
        message += "• Home field advantage weighting\n"
        
        return message

    def run_pipeline(self):
        """Execute full data fetching and projection pipeline"""
        print("\n" + "="*50)
        print("🚀 MLB Projection Pipeline Started")
        print("="*50 + "\n")
        
        print("📥 Step 1: Fetching game data from MLB StatsAPI...")
        fetcher = MLBDataFetcher()
        raw_data = fetcher.fetch_today_schedule()
        
        if raw_data.empty:
            print("\n⚠️ No games found. Pipeline halted.")
            self.send_telegram_message("⚾ *MLB Update*\nNo games scheduled for today. 🧢")
            return
        
        print(f"✅ Successfully fetched {len(raw_data)} games\n")
        
        print("🧮 Step 2: Generating advanced projections...")
        projector = MLBProjector()
        
        if hasattr(projector, 'run_projections'):
            projections_df = projector.run_projections()
        elif hasattr(projector, 'predict'):
            projections_df = projector.predict(raw_data)
        else:
            projections_df = raw_data
        
        if projections_df is None or projections_df.empty:
            print("\n⚠️ Projection generation failed.")
            self.send_telegram_message("⚠️ *MLB Error*\nProjections could not be generated.")
            return
        
        print(f"✅ Generated projections for {len(projections_df)} games\n")
        
        print("📤 Step 3: Formatting and sending results...")
        final_message = self.format_projections_message(projections_df)
        
        print("\n" + final_message + "\n")
        self.send_telegram_message(final_message)
        
        projections_df.to_csv(os.path.join(self.data_dir, 'projections.csv'), index=False)
        print(f"💾 Detailed projections saved to {self.data_dir}/projections.csv")
        
        print("\n" + "="*50)
        print("✅ Pipeline completed successfully!")
        print("="*50 + "\n")

if __name__ == "__main__":
    orchestrator = MLBOrchestrator()
    orchestrator.run_pipeline()
