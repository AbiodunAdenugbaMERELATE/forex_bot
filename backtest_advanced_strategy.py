import pandas as pd
import numpy as np
import math
from bot.strategies.advanced_strategy import AdvancedStrategy

HISTORICAL_CSV = 'EURUSD_3years.csv'  # Provide path to your historical data
START_BALANCE = 10000

class DummyAPI:
    def __init__(self, balance):
        self.balance = balance
    def get_account_summary(self):
        return {'balance': self.balance}
    # These aren't executed in backtest; live API handles real orders
    def place_order(self, *args, **kwargs):
        return {}
    def modify_trade_stops(self, *args, **kwargs):
        return {}

def run_backtest():
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    api = DummyAPI(START_BALANCE)
    strat = AdvancedStrategy(api, 'EUR_USD', {})
    strat.starting_equity = START_BALANCE
    strat.dry_run = True  # prevent any accidental live calls

    trades = []  # Each trade: dict(entry, side, units, sl, tp, exit, exit_reason, r, pnl)
    open_trade = None
    session_r = 0.0
    session_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['mid'])
        strat.data.append({'timestamp': row['timestamp'], 'mid': price})
        # Build rolling indicators similar to live
        if len(strat.data) < 120:
            continue
        sdf = pd.DataFrame(strat.data)
        sdf['SMA20'] = sdf['mid'].rolling(20).mean()
        sdf['SMA50'] = sdf['mid'].rolling(50).mean()
        sdf['STD20'] = sdf['mid'].rolling(20).std()
        if not strat.should_trade(sdf):
            # Still try trailing if open trade
            if open_trade and strat.trailing_enabled:
                # mimic R progress update for trailing decisions
                pass
            continue
        # Session reset simulation (UTC date boundary)
        cur_date = row['timestamp'].date()
        if session_date is None:
            session_date = cur_date
        if cur_date != session_date:
            session_r = 0.0
            session_date = cur_date
        # Session guard
        if strat.session_max_loss_r is not None and session_r <= strat.session_max_loss_r:
            continue
        if strat.session_max_profit_r is not None and session_r >= strat.session_max_profit_r:
            continue

        stop_pips = strat.calculate_dynamic_stop_loss(sdf)
        units = strat.calculate_position_size_volatility_adjusted(stop_pips, sdf)
        if units <= 0:
            continue
        pip = strat._pip_size()
        stop_distance = max(stop_pips, strat.min_stop_pips) * pip
        r_multiple = strat.default_tp_r_multiple
        if open_trade is None:
            # Entry conditions
            side = None
            if strat.should_buy(sdf):
                side = 'buy'
            elif strat.should_sell(sdf):
                side = 'sell'
            if side:
                entry = price
                if side == 'buy':
                    sl = entry - stop_distance
                    tp = entry + stop_distance * r_multiple
                else:
                    sl = entry + stop_distance
                    tp = entry - stop_distance * r_multiple
                trade = {
                    'entry': entry,
                    'side': side,
                    'units': units,
                    'sl': sl,
                    'tp': tp,
                    'risk_per_unit': abs(entry - sl),
                    'activated_trailing': False,
                    'locked_r': 0.0,
                    'timestamp': row['timestamp']
                }
                open_trade = trade
                trades.append(trade)
                continue
        else:
            # Manage open trade
            t = open_trade
            direction = 1 if t['side'] == 'buy' else -1
            # Check SL / TP hit (assume no intra-bar; use close as proxy)
            hit = None
            if t['side'] == 'buy':
                if price <= t['sl']:
                    hit = ('sl', t['sl'])
                elif price >= t['tp']:
                    hit = ('tp', t['tp'])
            else:
                if price >= t['sl']:
                    hit = ('sl', t['sl'])
                elif price <= t['tp']:
                    hit = ('tp', t['tp'])
            # Trailing simulation
            if hit is None and strat.trailing_enabled:
                risk = t['risk_per_unit']
                if risk and risk > 0:
                    per_unit_pl = (price - t['entry']) * direction
                    r_prog = per_unit_pl / risk
                    # Activate move to breakeven
                    if r_prog >= strat.trailing_activate_r:
                        be_buffer = strat.trailing_breakeven_buffer_pips * pip
                        desired_sl = t['entry'] + be_buffer * direction
                        if (t['side'] == 'buy' and desired_sl > t['sl']) or (t['side'] == 'sell' and desired_sl < t['sl']):
                            t['sl'] = desired_sl
                            t['activated_trailing'] = True
                    # Additional locking steps
                    if r_prog > strat.trailing_activate_r:
                        extra_r = r_prog - strat.trailing_activate_r
                        steps = int(extra_r / strat.trailing_step_r)
                        if steps > 0:
                            lock_r = min(strat.trailing_activate_r + steps * strat.trailing_step_r, strat.trailing_max_lock_r)
                            lock_distance = risk * (r_prog - lock_r)
                            if t['side'] == 'buy':
                                candidate = price - lock_distance
                                if candidate > t['sl']:
                                    t['sl'] = candidate
                            else:
                                candidate = price + lock_distance
                                if candidate < t['sl']:
                                    t['sl'] = candidate
            if hit is not None:
                reason, exit_price = hit
                per_unit_pl = (exit_price - t['entry']) * direction
                pnl = per_unit_pl * t['units']
                r_result = per_unit_pl / (t['risk_per_unit'] or 1)
                api.balance += pnl
                t.update({'exit': exit_price, 'exit_time': row['timestamp'], 'exit_reason': reason, 'pnl': pnl, 'r': r_result})
                session_r += r_result
                open_trade = None
                continue

    # Close any open trade at final price
    if open_trade:
        last_price = float(df['mid'].iloc[-1])
        t = open_trade
        direction = 1 if t['side'] == 'buy' else -1
        per_unit_pl = (last_price - t['entry']) * direction
        pnl = per_unit_pl * t['units']
        r_result = per_unit_pl / (t['risk_per_unit'] or 1)
        api.balance += pnl
        t.update({'exit': last_price, 'exit_time': df['timestamp'].iloc[-1], 'exit_reason': 'eod', 'pnl': pnl, 'r': r_result})

    # Analytics
    closed_trades = [t for t in trades if 'exit' in t]
    total_r = sum(t.get('r', 0) for t in closed_trades)
    wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losses = [t for t in closed_trades if t.get('pnl', 0) < 0]
    expectancy = total_r / len(closed_trades) if closed_trades else 0
    gross_win = sum(t.get('pnl', 0) for t in wins)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
    pf = (gross_win / gross_loss) if gross_loss else math.inf
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0

    print("=== Backtest Summary ===")
    print(f"Start Balance: {START_BALANCE}")
    print(f"End Balance:   {api.balance:.2f}")
    print(f"Closed Trades: {len(closed_trades)} | Wins: {len(wins)} | Losses: {len(losses)} | WinRate: {win_rate:.2f}%")
    print(f"Total R: {total_r:.2f} | Expectancy (R): {expectancy:.3f} | Profit Factor: {pf:.2f}")
    if closed_trades:
        avg_hold = np.mean([(t['exit_time'] - t['timestamp']).total_seconds()/3600 for t in closed_trades])
        print(f"Avg Hold (hrs): {avg_hold:.2f}")
    # Optional: dump CSV
    pd.DataFrame(closed_trades).to_csv('backtest_trades.csv', index=False)
    print("Trade details saved to backtest_trades.csv")

if __name__ == '__main__':
    run_backtest()
