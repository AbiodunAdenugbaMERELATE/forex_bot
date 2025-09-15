import sys
sys.path.append('.')  # Ensure root is in path for send_telegram import
try:
    from send_telegram import send_telegram_message
except ImportError:
    def send_telegram_message(msg):
        pass  # fallback if import fails
from bot.strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from scipy import stats
import logging
import time
from datetime import datetime, timedelta

class AdvancedStrategy(BaseStrategy):
    def __init__(self, api_client, instrument, strategies):
        super().__init__(api_client, instrument, strategies)
        self.data = []
        self.logger = logging.getLogger(__name__)
        
        # Risk Management Parameters - Adjusted for more trades
        self.max_risk_per_trade = 0.01  # 1% risk per trade (reduced from 2%)
        self.max_correlation_risk = 0.85  # Increased correlation threshold
        self.max_daily_drawdown = 0.05   # 5% max daily drawdown
        self.trailing_stop_pips = 30     # Reduced trailing stop
        
        # Market Regime Parameters - More sensitive
        self.volatility_window = 14      # Reduced from 20
        self.trend_strength_threshold = 20  # Reduced from 25
        self.regime_lookback = 50        # Reduced from 100
        
        # Performance Tracking
        self.trades = []
        self.daily_returns = []
        self.current_drawdown = 0
        
        # Market Adaptation
        self.regime = 'unknown'
        self.volatility_state = 'normal'
        self.last_optimization = time.time()
        self.optimization_interval = 1800  # 30 minutes (reduced from 1 hour)
        
        # Trade Cooldown
        self.last_trade_time = 0
        self.cooldown_seconds = 60  # 1 minute cooldown after each trade
        self.loss_streak = 0
        self.max_loss_streak = 3
        self.loss_streak_cooldown = 300  # 5 minutes cooldown after 3 consecutive losses
        
        # Risk Management Enhancements
        self.max_open_trades = 3
        self.equity_stop_pct = 0.8  # Stop trading if equity drops below 80% of starting
        self.starting_equity = None

        # --- Virtual Equity & Adaptive Sizing Controls ---
        import os
        self.virtual_equity = float(os.getenv("VIRTUAL_EQUITY", "0") or 0)
        self.virtual_equity_mode = os.getenv("VIRTUAL_EQUITY_MODE", "scale").lower()  # scale | cap | adaptive
        self.virtual_equity_initial = self.virtual_equity
        self.adapt_participation_pct = float(os.getenv("ADAPT_PARTICIPATION_PCT", "0.25"))
        self.adapt_max_multiplier = float(os.getenv("ADAPT_MAX_MULTIPLIER", "2.0"))
        self.adapt_cooldown = int(os.getenv("ADAPT_COOLDOWN_SECONDS", "3600"))
        self._last_adapt_ts = 0
        self._peak_real_balance = None

        # Risk override from env
        env_risk = os.getenv("MAX_RISK_PER_TRADE")
        if env_risk:
            try:
                self.max_risk_per_trade = float(env_risk)
            except ValueError:
                pass

        # Safety caps
        self.max_units_cap = int(os.getenv("MAX_UNITS_CAP", "75000"))
        self.min_stop_pips = int(os.getenv("MIN_STOP_PIPS_OVERRIDE", "15"))
        self.margin_usage_cap_pct = float(os.getenv("MARGIN_USAGE_CAP_PCT", "0.25"))
        self.account_currency = os.getenv("ACCOUNT_CURRENCY", "USD")
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

        # Diagnostics / adjustments
        self.enable_position_sizing_debug = True
        self.fixed_position_size = None
        self.retry_shrink_factors = [0.5, 0.25]
        self.last_rejection_time = None
        self.rejection_cooldown = 30

    def calculate_pip_value(self):
        """Approximate pip value per unit in account currency."""
        try:
            base, quote = self.instrument.split('_')
            pip_size = 0.01 if quote == 'JPY' else 0.0001
            if quote == self.account_currency:
                return pip_size
            price = float(self.data[-1]['mid']) if self.data else 1.0
            if self.account_currency == 'USD':
                return pip_size / price
            return pip_size
        except Exception:
            return 0.0001

    def get_effective_balance(self, real_balance: float) -> float:
        if real_balance <= 0:
            return real_balance
        mode = self.virtual_equity_mode
        if self.virtual_equity <= 0:
            return real_balance
        if mode == 'cap':
            return min(real_balance, self.virtual_equity)
        if mode == 'scale':
            if real_balance > self.virtual_equity:
                return self.virtual_equity
            return real_balance
        if mode == 'adaptive':
            if self._peak_real_balance is None:
                self._peak_real_balance = real_balance
            if real_balance > self._peak_real_balance:
                self._peak_real_balance = real_balance
            now = time.time()
            if now - self._last_adapt_ts > self.adapt_cooldown:
                target = self.virtual_equity_initial + (self._peak_real_balance - self.virtual_equity_initial) * self.adapt_participation_pct
                cap = self.virtual_equity_initial * self.adapt_max_multiplier
                target = max(self.virtual_equity_initial, min(target, cap))
                if target > (self.virtual_equity or 0):
                    self.logger.info(f"[AdaptiveVirtualEquity] Raising virtual equity {self.virtual_equity} -> {target}")
                    self.virtual_equity = target
                self._last_adapt_ts = now
            return min(real_balance, self.virtual_equity)
        return real_balance
        
    def calculate_position_size(self, stop_loss_pips):
        try:
            if stop_loss_pips < self.min_stop_pips:
                stop_loss_pips = self.min_stop_pips
            try:
                acct = self.api_client.get_account_summary()
            except Exception:
                acct = {}
            real_balance = float(acct.get('balance', self.starting_equity or 0) or 0)
            effective_balance = self.get_effective_balance(real_balance)
            if real_balance <= 0 or effective_balance <= 0:
                self.logger.warning(f"[PositionSizing] Non-positive balance real={real_balance} eff={effective_balance}")
                return 0
            pip_value = self.calculate_pip_value() or 0.0001
            risk_amount = effective_balance * self.max_risk_per_trade
            denom = stop_loss_pips * pip_value
            if denom <= 0:
                self.logger.warning(f"[PositionSizing] Invalid denominator stop={stop_loss_pips} pip_value={pip_value}")
                return 0
            raw_size = risk_amount / denom
            final_size = int(min(raw_size, self.max_units_cap))
            if self.enable_position_sizing_debug:
                self.logger.info(
                    f"[PositionSizing] real={real_balance:.2f} eff={effective_balance:.2f} risk%={self.max_risk_per_trade} "
                    f"riskAmt={risk_amount:.2f} stop={stop_loss_pips} pipVal={pip_value:.6f} raw={raw_size:.2f} final={final_size}"
                )
            return final_size
        except Exception as e:
            self.logger.error(f"[PositionSizing] Error: {e}")
            return 0
            
    def detect_market_regime(self, df):
        """Detect current market regime (trending, ranging, volatile)"""
        if len(df) < self.regime_lookback:
            return 'unknown'
            
        # Calculate ADX for trend strength
        df['TR'] = self.calculate_true_range(df)
        df['DM+'] = self.calculate_plus_dm(df)
        df['DM-'] = self.calculate_minus_dm(df)
        df['ADX'] = self.calculate_adx(df)
        
        # Calculate volatility
        df['volatility'] = df['mid'].rolling(self.volatility_window).std()
        current_vol = df['volatility'].iloc[-1]
        avg_vol = df['volatility'].mean()
        
        # Determine regime
        adx = df['ADX'].iloc[-1]
        if adx > self.trend_strength_threshold:
            if current_vol > avg_vol * 1.5:
                return 'volatile_trending'
            return 'trending'
        else:
            if current_vol > avg_vol * 1.5:
                return 'volatile_ranging'
            return 'ranging'
            
    def calculate_true_range(self, df):
        """Calculate True Range"""
        high = df['mid'].rolling(2).max()
        low = df['mid'].rolling(2).min()
        return high - low
        
    def calculate_plus_dm(self, df):
        """Calculate Plus Directional Movement"""
        diff = df['mid'].diff()
        return np.where(diff > 0, diff, 0)
        
    def calculate_minus_dm(self, df):
        """Calculate Minus Directional Movement"""
        diff = df['mid'].diff()
        return np.where(diff < 0, abs(diff), 0)
        
    def calculate_adx(self, df):
        """Calculate Average Directional Index"""
        smoothed_tr = df['TR'].rolling(14).mean()
        smoothed_plus_dm = df['DM+'].rolling(14).mean()
        smoothed_minus_dm = df['DM-'].rolling(14).mean()
        
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        return adx
        
    def check_correlation_risk(self):
        """Check correlation with other traded instruments"""
        correlations = {}
        for symbol, strategy in self.strategies.items():
            if symbol != self.instrument and len(strategy.data) > 0:
                other_prices = pd.DataFrame(strategy.data)['mid']
                our_prices = pd.DataFrame(self.data)['mid']
                if len(other_prices) == len(our_prices):
                    correlation = abs(other_prices.corr(our_prices))
                    correlations[symbol] = correlation
                    
        # Check if any correlation exceeds our threshold
        return any(corr > self.max_correlation_risk for corr in correlations.values())
        
    def optimize_parameters(self, df):
        """Periodically optimize strategy parameters based on recent market conditions"""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
        self.last_optimization = current_time
        # Run grid search optimization
        self.grid_search_optimize(df)
        # Adjust parameters based on market regime
        regime = self.detect_market_regime(df)
        volatility = df['mid'].rolling(self.volatility_window).std().iloc[-1]
        avg_volatility = df['mid'].rolling(self.volatility_window).std().mean()
        
        if regime == 'trending':
            self.trailing_stop_pips = 75
            self.max_risk_per_trade = 0.02
        elif regime == 'ranging':
            self.trailing_stop_pips = 30
            self.max_risk_per_trade = 0.015
        elif regime == 'volatile_trending':
            self.trailing_stop_pips = 100
            self.max_risk_per_trade = 0.01
        elif regime == 'volatile_ranging':
            self.trailing_stop_pips = 50
            self.max_risk_per_trade = 0.01
            
        # Adjust based on volatility
        if volatility > avg_volatility * 2:
            self.max_risk_per_trade *= 0.5

    def grid_search_optimize(self, df):
        """Simple grid search for SMA and stop loss parameters on recent data."""
        if len(df) < 200:
            return  # Not enough data
        best_params = None
        best_return = -float('inf')
        sma_fast_options = [10, 20, 30]
        sma_slow_options = [50, 100, 200]
        stop_loss_options = [20, 30, 50]
        for sma_fast in sma_fast_options:
            for sma_slow in sma_slow_options:
                if sma_fast >= sma_slow:
                    continue
                for stop_loss in stop_loss_options:
                    returns = self.simulate_strategy(df, sma_fast, sma_slow, stop_loss)
                    if returns > best_return:
                        best_return = returns
                        best_params = (sma_fast, sma_slow, stop_loss)
        if best_params:
            self.logger.info(f"Optimized params: SMA{best_params[0]}/{best_params[1]}, SL={best_params[2]}, Return={best_return:.2f}")
            self.sma_fast = best_params[0]
            self.sma_slow = best_params[1]
            self.optimized_stop_loss = best_params[2]
        else:
            self.sma_fast = 20
            self.sma_slow = 50
            self.optimized_stop_loss = 30

    def simulate_strategy(self, df, sma_fast, sma_slow, stop_loss):
        """Simulate simple SMA crossover strategy on recent data for optimization."""
        df = df.copy()
        df['SMA_FAST'] = df['mid'].rolling(window=sma_fast).mean()
        df['SMA_SLOW'] = df['mid'].rolling(window=sma_slow).mean()
        position = 0
        entry_price = 0
        pnl = 0
        for i in range(max(sma_fast, sma_slow), len(df)):
            if position == 0:
                if df['SMA_FAST'].iloc[i] > df['SMA_SLOW'].iloc[i] and df['SMA_FAST'].iloc[i-1] <= df['SMA_SLOW'].iloc[i-1]:
                    position = 1
                    entry_price = df['mid'].iloc[i]
                elif df['SMA_FAST'].iloc[i] < df['SMA_SLOW'].iloc[i] and df['SMA_FAST'].iloc[i-1] >= df['SMA_SLOW'].iloc[i-1]:
                    position = -1
                    entry_price = df['mid'].iloc[i]
            elif position == 1:
                # Check stop loss
                if df['mid'].iloc[i] < entry_price - stop_loss * 0.0001:
                    pnl += df['mid'].iloc[i] - entry_price
                    position = 0
                # Exit on cross
                elif df['SMA_FAST'].iloc[i] < df['SMA_SLOW'].iloc[i]:
                    pnl += df['mid'].iloc[i] - entry_price
                    position = 0
            elif position == -1:
                if df['mid'].iloc[i] > entry_price + stop_loss * 0.0001:
                    pnl += entry_price - df['mid'].iloc[i]
                    position = 0
                elif df['SMA_FAST'].iloc[i] > df['SMA_SLOW'].iloc[i]:
                    pnl += entry_price - df['mid'].iloc[i]
                    position = 0
        return pnl

    def on_price_update(self, data):
        try:
            mid_price = (float(data['bids'][0]['price']) + float(data['asks'][0]['price'])) / 2
            timestamp = data['time']
            self.data.append({'timestamp': timestamp, 'mid': mid_price})
            
            df = pd.DataFrame(self.data)
            # Set starting equity on first run
            if self.starting_equity is None:
                try:
                    account_info = self.api_client.get_account_summary()
                    self.starting_equity = float(account_info['balance'])
                except Exception:
                    self.starting_equity = 0
            if len(df) > 100:
                self.optimize_parameters(df)
                df['SMA20'] = df['mid'].rolling(window=20).mean()
                df['SMA50'] = df['mid'].rolling(window=50).mean()
                df['STD20'] = df['mid'].rolling(window=20).std()
                now = time.time()
                if now - self.last_trade_time < self.cooldown_seconds:
                    self.logger.info(f"In trade cooldown period. Skipping trade for {self.instrument}.")
                    return
                if self.loss_streak >= self.max_loss_streak and now - self.last_trade_time < self.loss_streak_cooldown:
                    self.logger.info(f"In loss streak cooldown period. Skipping trade for {self.instrument}.")
                    return
                # Equity stop
                try:
                    account_info = self.api_client.get_account_summary()
                    equity = float(account_info['balance'])
                    if self.starting_equity and equity < self.starting_equity * self.equity_stop_pct:
                        self.logger.warning(f"Equity stop triggered for {self.instrument}. Current equity: {equity}")
                        return
                except Exception:
                    pass
                # Max open trades check
                open_trades = self.get_open_trades_count()
                if open_trades >= self.max_open_trades:
                    self.logger.info(f"Max open trades reached for {self.instrument}. Skipping trade.")
                    return
                if self.should_trade(df):
                    stop_loss_pips = self.calculate_dynamic_stop_loss(df)
                    position_size = self.calculate_position_size_volatility_adjusted(stop_loss_pips, df)
                    if position_size > 0 and not self.check_correlation_risk():
                        if self.should_buy(df):
                            self.logger.info(f"Buy signal generated for {self.instrument} at {timestamp}")
                            self.place_sized_order('buy', position_size, timestamp)
                            self.last_trade_time = now
                        elif self.should_sell(df):
                            self.logger.info(f"Sell signal generated for {self.instrument} at {timestamp}")
                            self.place_sized_order('sell', position_size, timestamp)
                            self.last_trade_time = now
        except Exception as e:
            self.logger.error(f"Error processing price update: {e}")

    def should_trade(self, df):
        """Aggregate pre-trade checks. Returns True if strategy is allowed to evaluate entry signals.

        Checks include:
        - Sufficient data length for indicators
        - Not during high-impact news window
        - Market regime not unknown (after enough data)
        - Basic sanity on recent volatility (avoid NaNs)
        """
        try:
            # Need enough data for SMAs & momentum lookbacks
            if len(df) < 120:
                return False
            # Avoid NaNs in required columns
            required_cols = ['SMA20', 'SMA50', 'STD20']
            for col in required_cols:
                if df[col].isna().iloc[-1]:
                    return False
            # News filter
            if self.is_news_time():
                self.logger.info(f"Skipping trade due to news window for {self.instrument}")
                return False
            # Optional regime filter (only after lookback size)
            if len(df) >= self.regime_lookback:
                regime = self.detect_market_regime(df)
                if regime == 'unknown':
                    return False
            return True
        except Exception as e:
            self.logger.warning(f"should_trade check failed (defaulting False): {e}")
            return False
            
    def calculate_dynamic_stop_loss(self, df):
        """Calculate dynamic stop loss based on volatility"""
        atr = df['STD20'].iloc[-1]
        return max(20, int(atr * 100 * 1.5))  # Reduced minimum stop loss and multiplier
        

    def is_news_time(self):
        """Check if we're in a high-impact news window using FMP economic calendar API"""
        import os
        import requests
        from datetime import datetime, timedelta

        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            self.logger.warning("FMP_API_KEY not set. Skipping news filter.")
            return False

        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                # Detect legacy / deprecation message and permanently disable further calls this session
                txt = response.text
                if 'Legacy Endpoint' in txt or 'legacy' in txt.lower():
                    self.logger.warning("FMP economic calendar endpoint legacy notice. Disabling news filter for this run.")
                    self.is_news_time = lambda : False  # monkey-patch to skip future HTTP calls
                    return False
                self.logger.warning(f"FMP API error: {txt}")
                return False
            events = response.json()
            now = datetime.utcnow()
            window = timedelta(minutes=15)
            for event in events:
                # FMP returns 'date' as 'YYYY-MM-DD HH:MM:SS'
                event_time = datetime.strptime(event.get('date', ''), "%Y-%m-%d %H:%M:%S")
                # Only check high-impact events for the instrument's currency
                if abs((event_time - now).total_seconds()) < window.total_seconds():
                    if event.get('impact', '').lower() == 'high':
                        # Optionally, filter by currency
                        if self.instrument[:3] in event.get('currency', '') or self.instrument[4:] in event.get('currency', ''):
                            self.logger.info(f"High-impact news event detected: {event}")
                            return True
            return False
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return False
        
    def should_buy(self, df):
        """Enhanced buy signal with multiple confirmations"""
        # Trend following signal - using faster SMAs
        trend_signal = df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]
        
        # Volatility check - more permissive
        volatility_suitable = df['STD20'].iloc[-1] < df['STD20'].rolling(50).mean().iloc[-1] * 2
        
        # Price momentum - shorter lookback
        momentum = (df['mid'].iloc[-1] - df['mid'].iloc[-10]) / df['mid'].iloc[-10]
        momentum_positive = momentum > 0
        
        return trend_signal and volatility_suitable and momentum_positive
        
    def should_sell(self, df):
        """Enhanced sell signal with multiple confirmations"""
        # Trend following signal - using faster SMAs
        trend_signal = df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] >= df['SMA50'].iloc[-2]
        
        # Volatility check - more permissive
        volatility_suitable = df['STD20'].iloc[-1] < df['STD20'].rolling(50).mean().iloc[-1] * 2
        
        # Price momentum - shorter lookback
        momentum = (df['mid'].iloc[-1] - df['mid'].iloc[-10]) / df['mid'].iloc[-10]
        momentum_negative = momentum < 0
        
        return trend_signal and volatility_suitable and momentum_negative
    
    def handle_order_result(self, order_result, side, units, timestamp):
        """Handle order execution result, log errors, and check for slippage/rejection. Updates loss streak. Sends Telegram notifications."""
        strategy_name = "AdvancedStrategy"
        try:
            if not order_result:
                msg = f"❌ [{strategy_name}] Order result is empty for {self.instrument} at {timestamp}"
                self.logger.error(msg)
                send_telegram_message(msg)
                return
            if 'orderFillTransaction' in order_result:
                fill = order_result['orderFillTransaction']
                price = fill.get('price')
                pl = fill.get('pl')
                msg = f"✅ [{strategy_name}] Order filled: {side.upper()} {units} {self.instrument} at {price}, P/L: {pl} ({timestamp})"
                self.logger.info(msg)
                send_telegram_message(msg)
                # Update loss streak
                try:
                    pl_val = float(pl)
                    if pl_val < 0:
                        self.loss_streak += 1
                    else:
                        self.loss_streak = 0
                except Exception:
                    pass
            elif 'errorMessage' in order_result:
                msg = f"❌ [{strategy_name}] Order rejected for {self.instrument}: {order_result['errorMessage']} ({timestamp})"
                self.logger.error(msg)
                send_telegram_message(msg)
            else:
                msg = f"⚠️ [{strategy_name}] Order response for {self.instrument}: {order_result} ({timestamp})"
                self.logger.warning(msg)
                send_telegram_message(msg)
        except Exception as e:
            self.logger.error(f"Error handling order result: {e}")
            
    def get_open_trades_count(self):
        """Get the number of open trades for this instrument (stub, implement with API if needed)."""
        try:
            # If your API supports it, return the count of open trades for this instrument
            # Example: return self.api_client.get_open_trades_count(self.instrument)
            return 0  # Placeholder: always allow trades
        except Exception:
            return 0

    def calculate_position_size_volatility_adjusted(self, stop_loss_pips, df):
        """Adjust position size based on recent volatility with caps."""
        if self.fixed_position_size is not None:
            return self.fixed_position_size
        base_size = self.calculate_position_size(stop_loss_pips)
        if base_size <= 0:
            return 0
        try:
            recent_vol = df['STD20'].iloc[-1]
            avg_vol = df['STD20'].rolling(50).mean().iloc[-1]
            adjusted = base_size
            if recent_vol > avg_vol * 1.5:
                adjusted = int(base_size * 0.5)
            elif recent_vol < avg_vol * 0.7:
                adjusted = int(base_size * 1.2)
            if adjusted > self.max_units_cap:
                adjusted = self.max_units_cap
            if self.enable_position_sizing_debug:
                self.logger.info(f"[PositionSizing] volAdj base={base_size} final={adjusted} recentVol={recent_vol:.6f} avgVol={avg_vol:.6f}")
            return adjusted
        except Exception as e:
            self.logger.warning(f"[PositionSizing] Vol adjust error: {e}; using base.")
            return base_size

    def pre_check_margin(self, units):
        """Scale down units if projected margin usage exceeds cap."""
        try:
            acct = self.api_client.get_account_summary()
            margin_avail = float(acct.get('marginAvailable', 0))
            margin_rate = float(acct.get('marginRate', 0) or 0.02)
            if margin_avail <= 0:
                return units
            price = float(self.data[-1]['mid']) if self.data else 1.0
            required = abs(units) * price * margin_rate
            cap_allowed = margin_avail * self.margin_usage_cap_pct
            if required > cap_allowed and cap_allowed > 0:
                scale = cap_allowed / required
                new_units = int(units * scale)
                self.logger.info(f"[MarginCap] Reducing units {units}->{new_units} req={required:.2f} capAllowed={cap_allowed:.2f} rate={margin_rate}")
                return new_units
            return units
        except Exception:
            return units

    def place_sized_order(self, side, position_size, timestamp):
        if position_size <= 0:
            return
        units = position_size if side == 'buy' else -position_size
        units = self.pre_check_margin(units)
        if abs(units) < 1:
            self.logger.info("[Order] Units reduced below 1; skipping.")
            return
        if self.dry_run:
            msg = f"🧪 [AdvancedStrategy] DRY_RUN {side.upper()} {units} {self.instrument} ({timestamp})"
            self.logger.info(msg)
            send_telegram_message(msg)
            return
        order_result = self.api_client.place_order(self.instrument, units)
        self.handle_order_result(order_result, side, units, timestamp)