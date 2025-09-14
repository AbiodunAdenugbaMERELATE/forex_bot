import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot.strategies.advanced_strategy import AdvancedStrategy

# --- User must provide historical price data as a CSV file ---
# The CSV should have columns: timestamp, mid
# Example: EURUSD_3years.csv

HISTORICAL_CSV = 'EURUSD_3years.csv'  # <-- Change to your file
START_BALANCE = 10000

class DummyAPI:
    def get_account_summary(self):
        return {'balance': self.balance}
    def place_order(self, *args, **kwargs):
        return {'orderFillTransaction': {'price': kwargs.get('price', 0), 'pl': kwargs.get('pl', 0)}}

def run_backtest():
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=['timestamp'])
    api = DummyAPI()
    api.balance = START_BALANCE
    strategy = AdvancedStrategy(api, 'EUR_USD', {})
    strategy.starting_equity = START_BALANCE
    trades = []
    position = 0
    entry_price = 0
    for i in range(100, len(df)):
        # Simulate price update
        row = df.iloc[i]
        data = {'bids': [{'price': row['mid']}], 'asks': [{'price': row['mid']}], 'time': row['timestamp']}
        strategy.data.append({'timestamp': row['timestamp'], 'mid': row['mid']})
        # Use the same logic as in on_price_update, but simplified for backtest
        if len(strategy.data) > 100:
            sdf = pd.DataFrame(strategy.data)
            sdf['SMA20'] = sdf['mid'].rolling(window=20).mean()
            sdf['SMA50'] = sdf['mid'].rolling(window=50).mean()
            sdf['STD20'] = sdf['mid'].rolling(window=20).std()
            if strategy.should_trade(sdf):
                stop_loss_pips = strategy.calculate_dynamic_stop_loss(sdf)
                pos_size = strategy.calculate_position_size_volatility_adjusted(stop_loss_pips, sdf)
                if pos_size > 0 and not strategy.check_correlation_risk():
                    if position == 0:
                        if strategy.should_buy(sdf):
                            entry_price = row['mid']
                            position = 1
                            trades.append({'type': 'buy', 'entry': entry_price, 'timestamp': row['timestamp']})
                        elif strategy.should_sell(sdf):
                            entry_price = row['mid']
                            position = -1
                            trades.append({'type': 'sell', 'entry': entry_price, 'timestamp': row['timestamp']})
                    elif position == 1:
                        # Check for exit
                        if strategy.should_sell(sdf) or row['mid'] < entry_price - stop_loss_pips * 0.0001:
                            pl = row['mid'] - entry_price
                            api.balance += pl * pos_size
                            trades[-1]['exit'] = row['mid']
                            trades[-1]['pl'] = pl * pos_size
                            trades[-1]['exit_time'] = row['timestamp']
                            position = 0
                    elif position == -1:
                        if strategy.should_buy(sdf) or row['mid'] > entry_price + stop_loss_pips * 0.0001:
                            pl = entry_price - row['mid']
                            api.balance += pl * pos_size
                            trades[-1]['exit'] = row['mid']
                            trades[-1]['pl'] = pl * pos_size
                            trades[-1]['exit_time'] = row['timestamp']
                            position = 0
    # Summary
    total_pl = sum(t.get('pl', 0) for t in trades)
    win_trades = [t for t in trades if t.get('pl', 0) > 0]
    loss_trades = [t for t in trades if t.get('pl', 0) < 0]
    print(f"Backtest complete. Start balance: {START_BALANCE}")
    print(f"End balance: {api.balance:.2f}")
    print(f"Total P/L: {total_pl:.2f}")
    print(f"Total trades: {len(trades)} | Wins: {len(win_trades)} | Losses: {len(loss_trades)}")
    if trades:
        print(f"Win rate: {len(win_trades)/len(trades)*100:.2f}%")

if __name__ == "__main__":
    run_backtest()
