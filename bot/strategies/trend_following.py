import sys
sys.path.append('.')  # Ensure root is in path for send_telegram import
try:
    from send_telegram import send_telegram_message
except ImportError:
    def send_telegram_message(msg):
        pass  # fallback if import fails
from bot.strategies.base_strategy import BaseStrategy
import pandas as pd
import logging

class TrendFollowing(BaseStrategy):
    def __init__(self, api_client, instrument, strategies):
        super().__init__(api_client, instrument, strategies)
        self.data = []
        self.logger = logging.getLogger(__name__)

    def on_price_update(self, data):
        try:
            mid_price = (float(data['bids'][0]['price']) + float(data['asks'][0]['price'])) / 2
            timestamp = data['time']
            self.data.append({'timestamp': timestamp, 'mid': mid_price})

            df = pd.DataFrame(self.data)
            if len(df) > 200:
                df['SMA50'] = df['mid'].rolling(window=50).mean()
                df['SMA200'] = df['mid'].rolling(window=200).mean()
                if self.should_buy(df):
                    msg = f"[TrendFollowing] BUY {self.instrument} at {mid_price} ({timestamp})"
                    self.logger.info(f"Buy signal generated for {self.instrument} at {timestamp}")
                    send_telegram_message(msg)
                    self.api_client.place_order(self.instrument, 1000)
                elif self.should_sell(df):
                    msg = f"[TrendFollowing] SELL {self.instrument} at {mid_price} ({timestamp})"
                    self.logger.info(f"Sell signal generated for {self.instrument} at {timestamp}")
                    send_telegram_message(msg)
                    self.api_client.place_order(self.instrument, -1000)
        except Exception as e:
            self.logger.error(f"Error processing price update: {e}")

    def should_buy(self, df):
        return df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1] and df['SMA50'].iloc[-2] <= df['SMA200'].iloc[-2]

    def should_sell(self, df):
        return df['SMA50'].iloc[-1] < df['SMA200'].iloc[-1] and df['SMA50'].iloc[-2] >= df['SMA200'].iloc[-2]
