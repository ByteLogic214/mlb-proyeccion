import os
import sys
import requests
import pandas as pd

# Ensure the 'src' directory is in the python path for imports to work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import MLBDataFetcher
from predictor import MLBProjector

class MLBOrchestrator:
    def __init__(self):
        self.data_dir = "data"
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")

    def send_telegram_message(self, message):
        """Sends a formatted message to the configured Telegram bot."""
        if not self.telegram_token or not self.telegram_chat_id:
            print("Error: Telegram credentials not found in environment variables.")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("Telegram notification sent successfully.")
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

    def format_projections_message(self, df):
        """Formats the projection DataFrame into an aesthetic Telegram message."""
        if df.empty:
            return "⚾ *MLB Update*\nNo games scheduled for today. 🧢"

        message = "⚾ *MLB DAILY PROJECTIONS* ⚾\n"
        message += "━━━━━━━━━━━━━━━━━━\n\n"

        for _, row in df.iterrows():
            # Matchup and Venue
            message += f"🏟️ *{row['away_team']} @ {row['home_team']}*\n"
            
            # Pitchers and ERA
            message += (f"Pitchers: {row['away_sp']} ({row['away_sp_era']}) "
                        f"vs {row['home_sp']} ({row['home_sp_era']})\n")
            
            # Win Probabilities
            away_prob = row['prob_away_win'] * 100
            home_prob = row['prob_home_win'] * 100
            message += f"Win Prob: {row['away_team']} {away_prob:.0f}% | {row['home_team']} {home_prob:.0f}%\n"
            
            # Totals
            message += f"Total: {row['projected_total']} ☀️\n"
            message += "━━━━━━━━━━━━━━━━━━\n"

        return message

    def run_pipeline(self):
        """Executes the full ETL and Prediction pipeline."""
        print("Starting MLB Pipeline...")

        # 1. Extraction
        fetcher = MLBDataFetcher()
        raw_data = fetcher.fetch_today_schedule()

        if raw_data.empty:
            print("No games found today. Sending notification.")
            self.send_telegram_message("⚾ *MLB Update*\nNo games scheduled for today. 🧢")
            return

        # 2. Prediction
        print("Running projections...")
        projector = MLBProjector()
        projections_df = projector.run_projections()

        if projections_df is None or projections_df.empty:
            print("Projections failed or returned no data.")
            self.send_telegram_message("⚠️ *MLB Error*\nProjections could not be generated.")
            return

        # 3. Notification
        print("Formatting message and sending notification...")
        final_message = self.format_projections_message(projections_df)
        self.send_telegram_message(final_message)
        
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    orchestrator = MLBOrchestrator()
    orchestrator.run_pipeline()
