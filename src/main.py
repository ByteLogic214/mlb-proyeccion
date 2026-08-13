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

    def send_telegram_message(self, message):
        """Send formatted message via Telegram bot"""
        if not self.telegram_token or not self.telegram_chat_id:
            print("Warning: Telegram credentials not configured.")
            return False
            
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print("✅ Telegram notification sent successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False

    def format_projections_message(self, df):
        """Format projections into human-readable Telegram message"""
        if df is None or df.empty:
            return "⚾ *MLB Update*\nNo games scheduled for today. 🧢"

        message = "⚾ *MLB DAILY PROJECTIONS* ⚾\n"
        message += f"📅 {datetime.now().strftime('%B %d, %Y')}\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for idx, row in df.iterrows():
            # Matchup header
            message += f"🏟️ *Game {idx + 1}: {row.get('away_team', 'TBD')} @ {row.get('home_team', 'TBD')}*\n"
            
            # Pitcher information with ratings
            away_sp = row.get('away_sp', 'TBD')
            home_sp = row.get('home_sp', 'TBD')
            away_rating = row.get('away_sp_rating', 'N/A')
            home_rating = row.get('home_sp_rating', 'N/A')
            message += f"🥎 *Pitchers:* {away_sp} ({away_rating}) vs {home_sp} ({home_rating})\n"
            
            # Team ratings
            away_off = row.get('away_offense_rating', 'N/A')
            away_def = row.get('away_defense_rating', 'N/A')
            home_off = row.get('home_offense_rating', 'N/A')
            home_def = row.get('home_defense_rating', 'N/A')
            message += f"⚔️ *Ratings:* OFF/DEF\n"
            message += f"   • {row.get('away_team', 'TBD')}: {away_off}/{away_def}\n"
            message += f"   • {row.get('home_team', 'TBD')}: {home_off}/{home_def}\n"
            
            # Win probabilities
            away_prob = round(row.get('prob_away_win', 0) * 100)
            home_prob = round(row.get('prob_home_win', 0) * 100)
            message += f"🎯 *Win Probability:*\n"
            message += f"   • {row.get('away_team', 'TBD')}: {away_prob}%\n"
            message += f"   • {row.get('home_team', 'TBD')}: {home_prob}%\n"
            
            # Projected total
            total = row.get('projected_total', 'N/A')
            confidence = row.get('confidence', 'N/A')
            message += f"📊 *Total Runs:* {total} (Confidence: {confidence})\n"
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
        
        # Step 1: Fetch data
        print("📥 Step 1: Fetching game data from MLB StatsAPI...")
        fetcher = MLBDataFetcher()
        raw_data = fetcher.fetch_today_schedule()
        
        if raw_data.empty:
            print("\n⚠️ No games found. Pipeline halted.")
            self.send_telegram_message("⚾ *MLB Update*\nNo games scheduled for today. 🧢")
            return
        
        print(f"✅ Successfully fetched {len(raw_data)} games\n")
        
        # Step 2: Generate projections
        print("🧮 Step 2: Generating advanced projections...")
        projector = MLBProjector()
        projections_df = projector.run_projections()
        
        if projections_df is None or projections_df.empty:
            print("\n⚠️ Projection generation failed.")
            self.send_telegram_message("⚠️ *MLB Error*\nProjections could not be generated.")
            return
        
        print(f"✅ Generated projections for {len(projections_df)} games\n")
        
        # Step 3: Format and send results
        print("📤 Step 3: Formatting and sending results...")
        final_message = self.format_projections_message(projections_df)
        
        # Print to console
        print("\n" + final_message + "\n")
        
        # Send via Telegram
        self.send_telegram_message(final_message)
        
        # Save detailed results
        projections_df.to_csv(os.path.join(self.data_dir, 'projections.csv'), index=False)
        print(f"💾 Detailed projections saved to {self.data_dir}/projections.csv")
        
        print("\n" + "="*50)
        print("✅ Pipeline completed successfully!")
        print("="*50 + "\n")

if __name__ == "__main__":
    orchestrator = MLBOrchestrator()
    orchestrator.run_pipeline()
