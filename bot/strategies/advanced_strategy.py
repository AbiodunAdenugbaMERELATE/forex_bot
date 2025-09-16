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
        # Optional fixed unit sizing & long-only mode
        try:
            _env_fixed_units = os.getenv("FIXED_POSITION_SIZE") or os.getenv("FIXED_POSITION_UNITS")
            if _env_fixed_units:
                self.fixed_position_size = int(float(_env_fixed_units))
        except Exception:
            pass
        try:
            self.long_only = os.getenv("LONG_ONLY", "false").lower() == "true"
        except Exception:
            self.long_only = False

        # --- Adaptive risk contraction settings ---
        self.enable_adaptive_risk = True
        self.risk_reduction_factor = 0.5  # reduce risk% by 50% on trigger
        self.loss_streak_trigger = 3
        self.restore_step_factor = 0.25  # restore 25% of gap per winning trade
        self._base_max_risk_per_trade = self.max_risk_per_trade

        # --- Bracket / TP configuration (R-multiple take profit) ---
        import os as _os_brackets
        try:
            self.default_tp_r_multiple = float(_os_brackets.getenv("DEFAULT_TP_R_MULTIPLE", "1.5"))
        except Exception:
            self.default_tp_r_multiple = 1.5
        try:
            self.fallback_tp_r_multiple = float(_os_brackets.getenv("FALLBACK_TP_R_MULTIPLE", "1.0"))
        except Exception:
            self.fallback_tp_r_multiple = 1.0

        # --- Trade journaling ---
        self.journal_path = _os_brackets.getenv("TRADE_JOURNAL_PATH", "trade_log.csv")
        self._open_trades = {}  # trade_id -> info dict

        # --- Session risk guard configuration ---
        try:
            self.session_max_loss_r = float(_os_brackets.getenv("DAILY_MAX_LOSS_R", "-3"))  # negative threshold
        except Exception:
            self.session_max_loss_r = -3.0
        try:
            self.session_max_profit_r = float(_os_brackets.getenv("DAILY_MAX_PROFIT_R", "6"))
        except Exception:
            self.session_max_profit_r = 6.0
        try:
            self.session_reset_hour = int(_os_brackets.getenv("SESSION_RESET_HOUR", "0"))  # UTC hour
        except Exception:
            self.session_reset_hour = 0
        self.session_cum_realized_r = 0.0
        self.session_last_reset_date = None
        self.session_blocked = False

        # --- Spread / quality filter configuration ---
        try:
            self.max_spread_pips = float(_os_brackets.getenv("MAX_SPREAD_PIPS", "2.0"))
        except Exception:
            self.max_spread_pips = 2.0
        try:
            self.spread_lookback = int(_os_brackets.getenv("SPREAD_LOOKBACK", "50"))
        except Exception:
            self.spread_lookback = 50
        self._recent_spreads = []  # store raw spread in price units

        # --- Trailing stop configuration ---
        try:
            self.trailing_enabled = _os_brackets.getenv("TRAILING_ENABLED", "false").lower() == "true"
            self.trailing_activate_r = float(_os_brackets.getenv("TRAILING_ACTIVATE_R", "1.0"))  # R at which to start (move to BE)
            self.trailing_step_r = float(_os_brackets.getenv("TRAILING_STEP_R", "0.5"))  # Additional R increments to lock more
            self.trailing_max_lock_r = float(_os_brackets.getenv("TRAILING_MAX_LOCK_R", "5.0"))  # Cap locked R
            self.trailing_update_interval = int(_os_brackets.getenv("TRAILING_UPDATE_INTERVAL_SECONDS", "15"))  # Min seconds between update attempts
            self.trailing_min_move_pips = float(_os_brackets.getenv("TRAILING_MIN_MOVE_PIPS", "3"))  # Only modify if SL shift >= this
            self.trailing_breakeven_buffer_pips = float(_os_brackets.getenv("TRAILING_BREAKEVEN_BUFFER_PIPS", "1"))  # Small buffer past entry
        except Exception:
            self.trailing_enabled = False
            self.trailing_activate_r = 1.0
            self.trailing_step_r = 0.5
            self.trailing_max_lock_r = 5.0
            self.trailing_update_interval = 15
            self.trailing_min_move_pips = 3.0
            self.trailing_breakeven_buffer_pips = 1.0
        self._last_trailing_check = 0

        # --- Heartbeat / staleness configuration ---
        try:
            self.max_tick_staleness = int(_os_brackets.getenv("MAX_TICK_STALENESS_SECONDS", "30"))
        except Exception:
            self.max_tick_staleness = 30
        self._last_price_ts = None

        # --- ATR trailing configuration ---
        try:
            self.trailing_mode = _os_brackets.getenv("TRAILING_MODE", "R").upper()  # R | ATR
        except Exception:
            self.trailing_mode = "R"
        try:
            self.atr_period = int(_os_brackets.getenv("ATR_PERIOD", "14"))
        except Exception:
            self.atr_period = 14
        try:
            self.atr_multiplier = float(_os_brackets.getenv("ATR_MULTIPLIER", "2.0"))
        except Exception:
            self.atr_multiplier = 2.0

    # ---------------- Session Risk Guard Helpers ----------------
    def _maybe_reset_session_risk(self):
        """Reset session cumulative R at configured UTC hour boundary."""
        try:
            now_utc = datetime.utcnow()
            current_date = now_utc.date()
            if self.session_last_reset_date != current_date and now_utc.hour >= self.session_reset_hour:
                self.logger.info(f"[SessionRiskGuard] Resetting session metrics at {now_utc.isoformat()} UTC")
                self.session_cum_realized_r = 0.0
                self.session_blocked = False
                self.session_last_reset_date = current_date
        except Exception:
            pass

    def _compute_unrealized_r(self):
        """Approximate unrealized R across open trades (floating). Uses last mid price and stored risk_per_unit.

        Limitations: without continuous trade updates or partial fills detail, this is an approximation.
        """
        total_unrealized_r = 0.0
        if not self._open_trades:
            return 0.0
        try:
            current_price = float(self.data[-1]['mid']) if self.data else None
            if current_price is None:
                return 0.0
            for trade_id, tinfo in self._open_trades.items():
                entry = tinfo.get('entry')
                risk_per_unit = tinfo.get('risk_per_unit') or 0
                side = tinfo.get('side')
                units = tinfo.get('units') or 0
                if entry is None or risk_per_unit == 0:
                    continue
                direction = 1 if side == 'buy' else -1
                per_unit_pl = (current_price - entry) * direction
                r_progress = per_unit_pl / risk_per_unit if risk_per_unit else 0
                # Scale by proportion? R is per trade independent of size; keep raw R.
                total_unrealized_r += r_progress
        except Exception:
            return 0.0
        return total_unrealized_r

    # ---------------- Trailing Stop Helpers ----------------
    def _compute_trade_r_progress(self, trade_info, current_price):
        try:
            entry = trade_info.get('entry')
            sl = trade_info.get('sl')
            side = trade_info.get('side')
            risk_per_unit = trade_info.get('risk_per_unit')
            if None in (entry, sl) or not risk_per_unit or risk_per_unit == 0:
                return None
            direction = 1 if side == 'buy' else -1
            per_unit_pl = (current_price - entry) * direction
            r_progress = per_unit_pl / risk_per_unit
            return r_progress
        except Exception:
            return None

    def _maybe_trailing_update(self):
        """Iterate open trades and adjust stop losses based on R progress.

        Logic:
          1. Activate when R >= trailing_activate_r -> move stop to (entry +/- buffer)
          2. For each additional trailing_step_r gained, lock in an additional step of R, limited by trailing_max_lock_r.
          3. Only send modification if new SL differs by at least trailing_min_move_pips (avoid churn).
        """
        if not self.trailing_enabled:
            return
        if not self._open_trades:
            return
        now = time.time()
        if now - self._last_trailing_check < getattr(self, 'trailing_update_interval', 15):
            return
        self._last_trailing_check = now
        try:
            current_price = float(self.data[-1]['mid']) if self.data else None
            if current_price is None:
                return
            pip = self._pip_size()
            # Pre-compute ATR if ATR mode
            atr_value = None
            if self.trailing_mode == 'ATR' and len(self.data) >= self.atr_period + 2:
                try:
                    df_tmp = pd.DataFrame(self.data)
                    # True Range approximation with mid only
                    df_tmp['prev'] = df_tmp['mid'].shift(1)
                    df_tmp['rng'] = (df_tmp['mid'] - df_tmp['prev']).abs()
                    atr_value = df_tmp['rng'].rolling(self.atr_period).mean().iloc[-1]
                except Exception:
                    atr_value = None
            updates = 0
            for trade_id, info in list(self._open_trades.items()):
                r_prog = self._compute_trade_r_progress(info, current_price)
                if r_prog is None:
                    continue
                side = info.get('side')
                entry = info.get('entry')
                old_sl = info.get('sl')
                risk_per_unit = info.get('risk_per_unit')
                if None in (side, entry, old_sl, risk_per_unit) or risk_per_unit == 0:
                    continue
                desired_sl = None
                if self.trailing_mode == 'R':
                    # Activation: move to breakeven (+/- small buffer)
                    if r_prog >= self.trailing_activate_r:
                        be_buffer = self.trailing_breakeven_buffer_pips * pip
                        if side == 'buy':
                            desired_sl = entry + be_buffer  # lock a tiny gain
                        else:
                            desired_sl = entry - be_buffer
                    # Additional locking for progressed R beyond activation
                    if r_prog > self.trailing_activate_r and risk_per_unit > 0:
                        extra_r = r_prog - self.trailing_activate_r
                        steps = int(extra_r / self.trailing_step_r)
                        if steps > 0:
                            lock_r = min(self.trailing_activate_r + steps * self.trailing_step_r, self.trailing_max_lock_r)
                            lock_distance = risk_per_unit * (r_prog - lock_r)
                            if side == 'buy':
                                candidate = current_price - lock_distance
                            else:
                                candidate = current_price + lock_distance
                            if desired_sl is None:
                                desired_sl = candidate
                            else:
                                if side == 'buy':
                                    desired_sl = max(desired_sl, candidate)
                                else:
                                    desired_sl = min(desired_sl, candidate)
                elif self.trailing_mode == 'ATR' and atr_value:
                    # ATR-based trailing: stop = current_price -/+ atr_multiplier*ATR
                    buffer = self.atr_multiplier * atr_value
                    candidate = None
                    if side == 'buy':
                        candidate = current_price - buffer
                        # Never below original SL; ensure we only tighten
                        if candidate <= old_sl:
                            candidate = None
                    else:
                        candidate = current_price + buffer
                        if candidate >= old_sl:
                            candidate = None
                    if candidate is not None:
                        desired_sl = candidate if desired_sl is None else (max(desired_sl, candidate) if side == 'buy' else min(desired_sl, candidate))
                if desired_sl is None:
                    continue
                # Ensure the modification tightens stop (never loosens)
                if side == 'buy' and desired_sl <= old_sl:
                    continue
                if side == 'sell' and desired_sl >= old_sl:
                    continue
                # Movement threshold
                move_pips = abs(desired_sl - old_sl) / pip if pip else 0
                if move_pips < self.trailing_min_move_pips:
                    continue
                # Round precision
                desired_sl_rounded = round(desired_sl, 5)
                if self.dry_run:
                    self.logger.info(f"🧪 [Trailing] DRY_RUN modify SL trade={trade_id} {old_sl}->{desired_sl_rounded} Rprog={r_prog:.2f}")
                    self._journal_write('trailing_update', trade_id=trade_id, old_sl=old_sl, new_sl=desired_sl_rounded, r_prog=r_prog, dry_run=True)
                    info['sl'] = desired_sl_rounded
                    updates += 1
                    continue
                try:
                    resp = self.api_client.modify_trade_stops(trade_id, stop_loss=desired_sl_rounded)
                    info['sl'] = desired_sl_rounded
                    self.logger.info(f"[Trailing] Updated SL trade={trade_id} {old_sl}->{desired_sl_rounded} R={r_prog:.2f}")
                    self._journal_write('trailing_update', trade_id=trade_id, old_sl=old_sl, new_sl=desired_sl_rounded, r_prog=r_prog, api_resp=str(resp))
                    updates += 1
                except Exception as e:
                    self.logger.warning(f"[Trailing] Failed SL update trade={trade_id}: {e}")
            if updates:
                send_telegram_message(f"[Trailing] Updated {updates} stops {self.instrument}")
        except Exception as e:
            self.logger.warning(f"[Trailing] Error processing trailing updates: {e}")

    def _session_risk_guard_allows_trade(self):
        """Return True if session limits permit a new trade."""
        self._maybe_reset_session_risk()
        if self.session_blocked:
            return False
        unrealized_r = self._compute_unrealized_r()
        total_r = self.session_cum_realized_r + unrealized_r
        if self.session_max_loss_r is not None and total_r <= self.session_max_loss_r:
            self.logger.warning(f"[SessionRiskGuard] Trading blocked: totalR={total_r:.2f} <= maxLossR={self.session_max_loss_r}")
            self.session_blocked = True
            return False
        if self.session_max_profit_r is not None and total_r >= self.session_max_profit_r:
            self.logger.info(f"[SessionRiskGuard] Trading blocked: totalR={total_r:.2f} >= maxProfitR={self.session_max_profit_r}")
            self.session_blocked = True
            return False
        return True

    # ---------------- Journal Helpers ----------------
    def _journal_write(self, phase: str, **fields):
        import csv, time, os
        base = {
            'ts': time.time(),
            'phase': phase,
            'instrument': self.instrument,
            'regime': getattr(self, 'regime', None),
            'vol_state': getattr(self, 'volatility_state', None)
        }
        base.update(fields)
        file_exists = os.path.isfile(self.journal_path)
        try:
            with open(self.journal_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=base.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(base)
        except Exception as e:
            self.logger.warning(f"[Journal] Failed to write: {e}")

    def _refresh_regime_tags(self, df):
        """Update regime / volatility_state attributes just before journaling a trade."""
        try:
            if len(df) >= self.regime_lookback:
                # reuse detect_market_regime if available
                reg = self.detect_market_regime(df)
                self.regime = reg
                # Simple volatility state classification
                if 'volatility' in df.columns:
                    cur_vol = df['volatility'].iloc[-1]
                    avg_vol = df['volatility'].rolling(100).mean().iloc[-1] if len(df) >= 100 else df['volatility'].mean()
                    if avg_vol and cur_vol > avg_vol * 1.5:
                        self.volatility_state = 'high'
                    elif avg_vol and cur_vol < avg_vol * 0.7:
                        self.volatility_state = 'low'
                    else:
                        self.volatility_state = 'normal'
        except Exception:
            pass

    # ---------------- Bracket Helpers ----------------
    def _pip_size(self):
        """Return pip size for current instrument (basic heuristic)."""
        try:
            quote = self.instrument.split('_')[1]
            return 0.01 if quote == 'JPY' else 0.0001
        except Exception:
            return 0.0001

    def _compute_bracket_prices(self, side: str, entry_price: float, stop_loss_pips: int, r_multiple: float = None):
        """Compute absolute stop loss and take profit prices based on an R-multiple.

        side: 'buy' or 'sell'
        entry_price: current mid price
        stop_loss_pips: distance to stop in pips
        r_multiple: TP distance expressed in multiples of stop distance
        """
        if entry_price is None or stop_loss_pips is None:
            return None, None
        pip = self._pip_size()
        if stop_loss_pips < self.min_stop_pips:
            stop_loss_pips = self.min_stop_pips
        distance = stop_loss_pips * pip
        if not r_multiple or r_multiple <= 0:
            r_multiple = self.default_tp_r_multiple or 1.5
        if r_multiple <= 0:
            r_multiple = self.fallback_tp_r_multiple
        try:
            if side == 'buy':
                sl = entry_price - distance
                tp = entry_price + distance * r_multiple
            else:  # sell
                sl = entry_price + distance
                tp = entry_price - distance * r_multiple
            if sl <= 0 or tp <= 0:
                return None, None
            # OANDA FX price precision often 5 decimals (3 for JPY quote) – use 5 for safety
            return round(sl, 5), round(tp, 5)
        except Exception:
            return None, None

    def calculate_pip_value(self):
        """Approximate pip value per unit in account currency.

        Heuristic handling:
          - If quote currency equals account currency: pip value = pip_size
          - If account currency is USD and quote != USD: convert using 1/price heuristic (for XYZ_USD vs USD_XYZ)
          - If pair has USD as base (USD_JPY) and account != quote: pip value from pip_size * price (approx)
          - Fallback: pip_size

        Note: For rigorous accuracy you'd query conversion instrument (e.g. EUR_USD) for cross rates.
        """
        try:
            base, quote = self.instrument.split('_')
            pip_size = 0.01 if quote == 'JPY' else 0.0001
            if quote == self.account_currency:
                return pip_size
            price = float(self.data[-1]['mid']) if self.data else 1.0
            # If account currency is USD and USD not the quote, approximate via inverse
            if self.account_currency == 'USD':
                if quote != 'USD':
                    # instrument like EUR_JPY and account USD – we approximate by dividing pip in quote by USD value of quote
                    # Without cross rate, fallback to pip_size / price
                    return pip_size / price if price else pip_size
            # If instrument base is account currency (e.g., USD_JPY, account USD)
            if base == self.account_currency:
                # pip value scales with price for base=account
                return pip_size * price
            # Basic fallback
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
            # Record last tick time (epoch)
            try:
                self._last_price_ts = time.time()
            except Exception:
                pass
            
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
                    # Staleness guard: if last tick older than threshold, skip new entries
                    if self._last_price_ts is not None:
                        age = time.time() - self._last_price_ts
                        if age > self.max_tick_staleness:
                            self.logger.warning(f"[Heartbeat] Stale feed age={age:.1f}s > {self.max_tick_staleness}s; skipping entries")
                            return
                    # Spread quality filter update
                    try:
                        # Derive latest spread from last stored tick if present
                        latest_spread = None
                        if self.data and 'spread' in self.data[-1]:
                            latest_spread = self.data[-1].get('spread')
                        if latest_spread is not None:
                            self._recent_spreads.append(float(latest_spread))
                            if len(self._recent_spreads) > self.spread_lookback:
                                self._recent_spreads.pop(0)
                            pip = self._pip_size()
                            current_spread = float(latest_spread)
                            spread_pips = current_spread / pip if pip else 0
                            spread_ok = True
                            if spread_pips > self.max_spread_pips:
                                spread_ok = False
                            if spread_ok and len(self._recent_spreads) >= 10:
                                import statistics
                                med = statistics.median(self._recent_spreads)
                                if current_spread > med * 3:
                                    spread_ok = False
                            if not spread_ok:
                                self.logger.info(f"[SpreadFilter] Skipping trade spread={spread_pips:.2f}p > limit={self.max_spread_pips}p")
                                return
                    except Exception:
                        pass
                    # Session risk guard check BEFORE sizing
                    if not self._session_risk_guard_allows_trade():
                        self.logger.info(f"[SessionRiskGuard] Skipping entry for {self.instrument} due to session limits.")
                        return
                    stop_loss_pips = self.calculate_dynamic_stop_loss(df)
                    position_size = self.calculate_position_size_volatility_adjusted(stop_loss_pips, df)
                    if position_size > 0 and not self.check_correlation_risk():
                        if self.should_buy(df):
                            self.logger.info(f"Buy signal generated for {self.instrument} at {timestamp}")
                            self.place_sized_order('buy', position_size, timestamp, stop_loss_pips)
                            self.last_trade_time = now
                        elif (not getattr(self, 'long_only', False)) and self.should_sell(df):
                            self.logger.info(f"Sell signal generated for {self.instrument} at {timestamp}")
                            self.place_sized_order('sell', position_size, timestamp, stop_loss_pips)
                            self.last_trade_time = now
                # After processing potential trade entries, attempt trailing updates
                self._maybe_trailing_update()
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
    
    def handle_order_result(self, order_result, side, units, timestamp, intended_sl=None, intended_tp=None):
        """Handle order response; extended to log bracket details if present."""
        strategy_name = "AdvancedStrategy"
        try:
            if not order_result:
                msg = f"❌ [{strategy_name}] Empty order result {self.instrument} {timestamp}"
                self.logger.error(msg)
                send_telegram_message(msg)
                self._journal_write('rejected', side=side, units=units, reason='empty_response')
                return
            if 'orderFillTransaction' in order_result:
                fill = order_result['orderFillTransaction']
                price = fill.get('price')
                pl = fill.get('pl')
                # Extract bracket info if echoed back
                sl = None
                tp = None
                try:
                    if 'stopLossOnFill' in fill:
                        sl = fill['stopLossOnFill'].get('price')
                    if 'takeProfitOnFill' in fill:
                        tp = fill['takeProfitOnFill'].get('price')
                except Exception:
                    pass
                # Some OANDA responses include separate transactions for SL/TP
                try:
                    if not sl and 'stopLossOrderTransaction' in order_result:
                        sl = order_result['stopLossOrderTransaction'].get('price')
                    if not tp and 'takeProfitOrderTransaction' in order_result:
                        tp = order_result['takeProfitOrderTransaction'].get('price')
                except Exception:
                    pass
                trade_id = None
                try:
                    trade_opened = fill.get('tradeOpened') or {}
                    trade_id = trade_opened.get('tradeID') or trade_opened.get('id')
                except Exception:
                    trade_id = None
                # Fallback to intended values if broker did not echo bracket
                if (sl is None or tp is None) and (intended_sl is not None or intended_tp is not None):
                    if sl is None:
                        sl = intended_sl
                    if tp is None:
                        tp = intended_tp
                # Enhanced direction & visual formatting
                direction_emoji = '🟢' if side == 'buy' else '🔴'
                side_label = 'LONG' if side == 'buy' else 'SHORT'
                msg = (f"{direction_emoji} ENTRY [{strategy_name}] {side_label} {abs(units)} {self.instrument} @ {price} "
                       f"SL={sl} TP={tp} P/L:{pl} ({timestamp})")
                self.logger.info(msg)
                send_telegram_message(msg)
                # Track open trade for future trailing or R computation
                if trade_id:
                    try:
                        entry_price_f = float(price) if price else None
                        sl_f = float(sl) if sl else None
                        risk_per_unit = None
                        if entry_price_f is not None and sl_f is not None:
                            risk_per_unit = abs(entry_price_f - sl_f)
                        self._open_trades[trade_id] = {
                            'side': side,
                            'units': units,
                            'entry': entry_price_f,
                            'sl': sl_f,
                            'tp': float(tp) if tp else None,
                            'risk_per_unit': risk_per_unit,
                            'timestamp': timestamp
                        }
                    except Exception:
                        pass
                self._journal_write('filled', side=side, units=units, price=price, sl=sl, tp=tp, trade_id=trade_id, pl=pl)
                # If still missing protective orders, attempt to attach them now
                try:
                    if trade_id and (sl is None or tp is None) and (intended_sl or intended_tp):
                        attach_sl = intended_sl if sl is None else None
                        attach_tp = intended_tp if tp is None else None
                        if attach_sl or attach_tp:
                            r = self.api_client.modify_trade_stops(trade_id, stop_loss_price=attach_sl, take_profit_price=attach_tp)
                            self.logger.info(f"[BracketAttach] Attempted attach SL={attach_sl} TP={attach_tp} resp={r}")
                            send_telegram_message(f"🔧 [AdvancedStrategy] Added missing protective orders trade {trade_id} SL={attach_sl} TP={attach_tp}")
                            # Update local record if success heuristically
                            ot = self._open_trades.get(trade_id)
                            if ot:
                                if attach_sl and ot.get('sl') is None:
                                    ot['sl'] = attach_sl
                                if attach_tp and ot.get('tp') is None:
                                    ot['tp'] = attach_tp
                except Exception as _e:
                    self.logger.warning(f"[BracketAttach] failed: {_e}")
                # Adaptive risk contraction: treat negative pl as loss
                try:
                    if self.enable_adaptive_risk and pl is not None:
                        pnl_val = float(pl)
                        if pnl_val < 0:
                            self.loss_streak += 1
                            if self.loss_streak >= self.loss_streak_trigger:
                                # Reduce risk
                                new_risk = self.max_risk_per_trade * self.risk_reduction_factor
                                # Ensure not below a minimal floor (optional)
                                floor = 0.0005  # 0.05%
                                new_risk = max(floor, new_risk)
                                if new_risk < self.max_risk_per_trade:
                                    self.logger.info(f"[AdaptiveRisk] Reducing max_risk_per_trade {self.max_risk_per_trade:.4f} -> {new_risk:.4f} after streak={self.loss_streak}")
                                    self.max_risk_per_trade = new_risk
                        else:  # win resets / partial restoration
                            if self.loss_streak > 0:
                                self.loss_streak = 0
                            if self.max_risk_per_trade < self._base_max_risk_per_trade:
                                gap = self._base_max_risk_per_trade - self.max_risk_per_trade
                                restore = gap * self.restore_step_factor
                                new_risk = min(self._base_max_risk_per_trade, self.max_risk_per_trade + restore)
                                if new_risk > self.max_risk_per_trade:
                                    self.logger.info(f"[AdaptiveRisk] Restoring max_risk_per_trade {self.max_risk_per_trade:.4f} -> {new_risk:.4f}")
                                    self.max_risk_per_trade = new_risk
                except Exception:
                    pass
            elif 'orderCancelTransaction' in order_result and 'errorMessage' in order_result:
                err = order_result.get('errorMessage')
                msg = f"❌ [{strategy_name}] Rejected {self.instrument}: {err} ({timestamp})"
                self.logger.error(msg)
                send_telegram_message(msg)
                self._journal_write('rejected', side=side, units=units, reason=err)
            else:
                msg = f"⚠️ [{strategy_name}] Unexpected order response {self.instrument}: {order_result} ({timestamp})"
                self.logger.warning(msg)
                send_telegram_message(msg)
                self._journal_write('other', side=side, units=units, raw=str(order_result))
        except Exception as e:
            self.logger.error(f"Error handling order result: {e}")

    def _notify_trade_closed(self, trade_id, info, close_price, reason):
        """Send a visually rich Telegram notification for a closed trade.

        reason: 'tp' | 'sl' | 'profit' | 'loss' | 'unknown'
        """
        try:
            side = info.get('side')
            units = abs(info.get('units', 0))
            entry = info.get('entry')
            risk_per_unit = info.get('risk_per_unit') or 0
            sl = info.get('sl')
            tp = info.get('tp')
            direction_emoji = '🟢' if side == 'buy' else '🔴'
            side_label = 'LONG' if side == 'buy' else 'SHORT'
            outcome_icon = '✅'
            if reason in ('sl', 'loss'):
                outcome_icon = '🛑'
            elif reason == 'tp':
                outcome_icon = '✅'
            elif reason == 'unknown':
                outcome_icon = '⚠️'
            r_result = 0.0
            pl_value = 0.0
            if entry is not None and close_price is not None and risk_per_unit:
                direction = 1 if side == 'buy' else -1
                per_unit_pl = (close_price - entry) * direction
                pl_value = per_unit_pl * units
                r_result = per_unit_pl / risk_per_unit if risk_per_unit else 0.0
            # Classify if not explicitly determined
            if reason == 'unknown':
                if r_result > 0.05:
                    outcome_icon = '✅'
                    reason = 'profit'
                elif r_result < -0.05:
                    outcome_icon = '🛑'
                    reason = 'loss'
            msg = (f"{outcome_icon} CLOSE {direction_emoji} {side_label} {units} {self.instrument} "
                   f"@ {close_price} R={r_result:.2f} P/L={pl_value:.2f} ({reason.upper()})")
            self.logger.info(msg)
            send_telegram_message(msg)
            # Journal closure summary (phase)
            self._journal_write('closed', side=side, units=units, price=close_price, r=r_result, reason=reason, trade_id=trade_id)
        except Exception as e:
            self.logger.warning(f"[CloseNotify] error: {e}")
            
    def get_open_trades_count(self):
        """Get real open trade count for this instrument and reconcile local state.

        Reconciliation: remove any _open_trades entries whose trade_id no longer appears.
        """
        try:
            count = 0
            api_count = 0
            if hasattr(self.api_client, 'get_open_trades_count'):
                api_count = self.api_client.get_open_trades_count(self.instrument)
                count = api_count
            # Attempt reconciliation if API provides list (future enhancement could add list method)
            # If counts mismatch vs local, clear orphaned trades (approximation)
            if self._open_trades and api_count == 0:
                # Assume all locally tracked trades are closed; attempt classification and notify
                last_price = None
                try:
                    if self.data:
                        last_price = float(self.data[-1]['mid'])
                except Exception:
                    last_price = None
                for tid, info in list(self._open_trades.items()):
                    try:
                        entry = info.get('entry')
                        tp = info.get('tp')
                        sl = info.get('sl')
                        risk = info.get('risk_per_unit') or 0
                        side = info.get('side')
                        close_price = last_price if last_price is not None else entry
                        reason = 'unknown'
                        if close_price is not None:
                            tol = 0.0002  # generic tolerance; could refine via pip size
                            if tp and abs(close_price - tp) <= tol:
                                reason = 'tp'
                            elif sl and abs(close_price - sl) <= tol:
                                reason = 'sl'
                        # compute R add to session
                        if entry is not None and close_price is not None and risk:
                            direction = 1 if side == 'buy' else -1
                            per_unit_pl = (close_price - entry) * direction
                            r_result = per_unit_pl / risk if risk else 0
                            self.session_cum_realized_r += r_result
                        self._notify_trade_closed(tid, info, close_price, reason)
                    except Exception:
                        pass
                self._open_trades.clear()
            return count
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

    def place_sized_order(self, side, position_size, timestamp, stop_loss_pips=None):
        if position_size <= 0:
            return
        units = position_size if side == 'buy' else -position_size
        units = self.pre_check_margin(units)
        if abs(units) < 1:
            self.logger.info("[Order] Units reduced below 1; skipping.")
            return
        # Determine entry price (mid of last tick)
        entry_price = None
        try:
            if self.data:
                entry_price = float(self.data[-1]['mid'])
        except Exception:
            entry_price = None

        sl_price = tp_price = None
        if stop_loss_pips is not None and entry_price is not None:
            sl_price, tp_price = self._compute_bracket_prices(side, entry_price, int(stop_loss_pips))
        if self.enable_position_sizing_debug:
            self.logger.info(f"[BracketPrep] side={side} units={units} stop_pips={stop_loss_pips} SL={sl_price} TP={tp_price}")
        # Refresh regime tags before journaling/submitting
        try:
            df_tmp = None
            if self.data and len(self.data) >= 20:
                import pandas as _pd_tmp
                df_tmp = _pd_tmp.DataFrame(self.data)
                if 'volatility' not in df_tmp.columns and len(df_tmp) >= self.volatility_window:
                    df_tmp['volatility'] = df_tmp['mid'].rolling(self.volatility_window).std()
            if df_tmp is not None:
                self._refresh_regime_tags(df_tmp)
        except Exception:
            pass

        if self.dry_run:
            msg = f"🧪 [AdvancedStrategy] DRY_RUN {side.upper()} {units} {self.instrument} SL={sl_price} TP={tp_price} ({timestamp})"
            self.logger.info(msg)
            send_telegram_message(msg)
            self._journal_write('submitted', side=side, units=units, dry_run=True, sl=sl_price, tp=tp_price,
                                stop_pips=stop_loss_pips)
            return
        order_result = self.api_client.place_order(
            self.instrument,
            units,
            order_type='MARKET',
            stop_loss_price=sl_price,
            take_profit_price=tp_price
        )
        self._journal_write('submitted', side=side, units=units, dry_run=False, sl=sl_price, tp=tp_price,
                            stop_pips=stop_loss_pips)
        self.handle_order_result(order_result, side, units, timestamp, intended_sl=sl_price, intended_tp=tp_price)