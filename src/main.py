import os
import sys
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import MLBDataFetcher
from predictor import MLBProjector

class MLBOrchestrator:
    def __init__(self):
        self.data_dir = "data"
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def send_telegram_message(self, message):
        if not self.telegram_token or not self.telegram_chat_id:
            print("Error: Telegram credentials missing.")
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload).raise_for_status()
            print("Telegram notification sent.")
        except Exception as e:
            print(f"Failed to send Telegram: {e}")

    def format_projections_message(self, df):
        if df.empty:
            return "⚾ *MLB Update*\nNo games scheduled for today. 🧢"

        message = "⚾ *MLB DAILY PROJECTIONS* ⚾\n"
        message += "━━━━━━━━━━━━━━━━━━\n\n"

        for _, row in df.iterrows():
            message += f"🏟️ *{row['away_team']} @ {row['home_team']}*\n"
            
            # FIX: Added .get() to prevent KeyError if columns are missing
            a_sp = row.get('away_sp', 'TBD')
            a_era = row.get('away_sp_era', 'N/A')
            h_sp = row.get('home_sp', 'TBD')
            h_era = row.get('home_sp_era', 'N/A')
            
            message += f"Pitchers: {a_sp} ({a_era}) vs {h_sp} ({h_era})\n"
            
            away_prob = row['prob_away_win'] * 100
            home_prob = row['prob_home_win'] * 100
            message += f"Win Prob: {row['away_team']} {away_prob:.0f}% | {row['home_team']} {home_prob:.0f}%\n"
            message += f"Total: {row['projected_total']} ☀️\n"
            message += "━━━━━━━━━━━━━━━━━━\n"

        return message

    def run_pipeline(self):
        print("Starting MLB Pipeline...")
        fetcher = MLBDataFetcher()
        raw_data = fetcher.fetch_today_schedule()

        if raw_data.empty:
            self.send_telegram_message("⚾ *MLB Update*\nNo games scheduled for today. 🧢")
            return

        print("Running projections...")
        projector = MLBProjector()
        projections_df = projector.run_projections()

        if projections_df is None or projections_df.empty:
            self.send_telegram_message("⚠️ *MLB Error*\nProjections could not be generated.")
            return

        print("Sending notification...")
        final_message = self.format_projections_message(projections_df)
        self.send_telegram_message(final_message)
        print("Pipeline completed successfully.")

if __name__ == "__main__":
    orchestrator = MLBOrchestrator()
    orchestrator.run_pipeline()
