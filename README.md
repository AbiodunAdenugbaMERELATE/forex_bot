<div align="center">

# Forex Trading Bot

Robust, modular, event‑driven OANDA Forex trading bot with layered risk controls, adaptive capital scaling, bracket orders, trailing stops, journaling & backtest parity.

</div>

---

## 1. Overview

This project automates discretionary‑style position management with systematic discipline:

1. Streams real‑time prices from OANDA
2. Applies strategy logic (trend / regime + filters)
3. Evaluates layered risk & quality gates (spread, session R, loss streak cooling, heartbeat)
4. Sizes the trade (virtual equity + risk % + safety caps OR fixed units)
5. Submits market order with attached Stop Loss & Take Profit (bracket)
6. Optionally manages a trailing stop (R or ATR mode)
7. Journals every phase (submitted, filled, trailing update, rejection)
8. Notifies via Telegram

Goal: Controlled growth with capital protection while preventing account blow‑ups due to hidden compounding or adverse execution conditions.

---

## 2. High‑Level Architecture

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Broker API | `bot/api/oanda_api.py` | REST order placement, stop modification, account summary, open trade sync |
| Streaming | `bot/api/websocket_client.py` | Live price ticks feeding strategy loop |
| Strategy Core | `bot/strategies/advanced_strategy.py` | Signal evaluation, gating, sizing, order orchestration, trailing, journaling |
| Risk Layer | Embedded in strategy | Virtual equity, risk %, margin cap, min stop, session R guard, spread filter, heartbeat |
| Analytics | `journal_summary.py` | Post‑trade metrics (win rate, PF, expectancy) |
| Backtest | `backtest_advanced_strategy.py` | Simulates SL/TP, trailing, R outcomes |
| Messaging | `send_telegram.py` | Telegram push notifications |
| Config | `.env` | Behavior & parameter toggles |

---

## 3. Feature Stack (Risk & Control Layers)

| Category | Controls |
|----------|----------|
| Capital Scaling | Virtual equity modes (scale / cap / adaptive) |
| Position Sizing | Risk % of effective balance, min stop pips, margin usage cap, max units cap, optional fixed units |
| Protective Orders | Immediate bracket SL/TP with R‑multiple target selection |
| Trailing | R‑based progressive locking OR ATR volatility trailing |
| Session Guard | Daily cumulative R loss/profit thresholds + timed reset |
| Market Quality | Spread ceiling + outlier suppression using rolling median |
| Execution Integrity | Heartbeat / tick staleness detection |
| Adaptive Risk | Risk contraction after consecutive losses (hooks in place) |
| Direction Filter | Optional long‑only mode |
| Journaling | Detailed CSV with timestamps, regime, volatility & R progression |

---

## 4. Execution Flow (Tick → Decision → Management)

1. Receive tick (bid/ask/mid). Record timestamp.
2. Update rolling indicators (trend/regime, volatility, spread window, ATR if enabled).
3. Quality gates: heartbeat ok? spread within limit? session R within envelope? not cooling down? market open? 
4. Strategy signal check (e.g., trend alignment). Long‑only filter may reject shorts.
5. Compute stop distance (bounded by `MIN_STOP_PIPS_OVERRIDE`).
6. Position sizing: (unless `FIXED_POSITION_SIZE`) risk fraction * effective balance / (pip value * stop pips) → apply margin & unit caps.
7. Attach bracket: SL price, TP via `DEFAULT_TP_R_MULTIPLE` fallback logic.
8. Submit order; on fill update journal & local open trade map.
9. Periodically evaluate trailing logic (R or ATR mode) → adjust stop if threshold advanced.
10. On TP/SL closure (or reconciliation) update realized R and session metrics.

---

## 5. Strategy Logic (Conceptual Summary)

The included advanced strategy combines:
* Trend / regime filters (e.g., volatility / directional bias heuristics)
* Spread & heartbeat gating for execution quality
* Session risk envelope to halt runaway loss days or lock in profit
* Optional trailing to harvest extended moves while capping give‑back

You can extend signals by adding new condition blocks inside `on_price_update` before the sizing / order section.

---

## 6. Environment Variables (Configuration Reference)

All variables are read at initialization; restart the bot after changes. Omitted variables fall back to default values embedded below.

### 6.1 Core Auth / Integrations
| Variable | Required | Description |
|----------|----------|-------------|
| OANDA_API_KEY | Yes | OANDA REST & streaming access token. |
| OANDA_ACCOUNT_ID | Yes | OANDA account (practice or live) identifier. |
| TELEGRAM_TOKEN | Optional (Yes if notifications desired) | Telegram bot API token. |
| TELEGRAM_CHAT_ID | Optional | Destination chat/user ID for notifications. |
| FMP_API_KEY | Optional | Enables news filter (skip trades around events). |

### 6.2 Risk & Capital Scaling
| Variable | Default | Effect |
|----------|---------|--------|
| MAX_RISK_PER_TRADE | 0.01 | Fraction of effective balance risked per trade (ignored if `FIXED_POSITION_SIZE`). |
| VIRTUAL_EQUITY | 0 / (ex: 2000) | If > 0 modifies effective balance baseline. |
| VIRTUAL_EQUITY_MODE | scale | scale: pretend balance = min(real, virtual). cap: treat balance = min(real, virtual) but never exceed. adaptive: grows virtual equity gradually using participation pct & multipliers. |
| ADAPT_PARTICIPATION_PCT | 0.25 | Portion of real balance considered when adaptive growth triggers. |
| ADAPT_MAX_MULTIPLIER | 2.0 | Upper limit on adaptive virtual equity expansion vs initial. |
| ADAPT_COOLDOWN_SECONDS | 3600 | Time between adaptive expansions. |
| MIN_STOP_PIPS_OVERRIDE | 15 | Floors stop distance to avoid unrealistically tight stops. |
| MARGIN_USAGE_CAP_PCT | 0.25 | Rejects trade if projected margin usage exceeds this ratio. |
| MAX_UNITS_CAP | 75000 | Hard absolute units ceiling. |
| ACCOUNT_CURRENCY | USD | Base currency for pip value heuristics. |
| DRY_RUN | false | If true, no orders sent; logs sizing & intent only. |
| EQUITY_STOP_PCT | (commented) | Optional circuit breaker: pause if drawdown ≥ this percent. |

### 6.3 Position Sizing Overrides
| Variable | Default | Effect |
|----------|---------|--------|
| FIXED_POSITION_SIZE | (unset) | When set, completely bypass dynamic risk sizing. |
| FIXED_POSITION_UNITS | (alias) | Legacy alias; first non‑empty used. |
| LONG_ONLY | false | If true, short signals are ignored. |

### 6.4 Bracket & Take Profit
| Variable | Default | Effect |
|----------|---------|--------|
| DEFAULT_TP_R_MULTIPLE | 1.5 | Primary R multiple for TP distance. |
| FALLBACK_TP_R_MULTIPLE | 1.0 | Used if primary computation fails. |
| TRADE_JOURNAL_PATH | trade_log.csv | CSV output path for trade journal. |

### 6.5 Session & Daily Guards
| Variable | Default | Effect |
|----------|---------|--------|
| DAILY_MAX_LOSS_R | -3 | Stop opening new trades beyond this cumulative R loss. |
| DAILY_MAX_PROFIT_R | 6 | Lock in gains; halt new entries once exceeded. |
| SESSION_RESET_HOUR | 0 | UTC hour to reset session metrics. |

### 6.6 Spread & Market Quality
| Variable | Default | Effect |
|----------|---------|--------|
| MAX_SPREAD_PIPS | 2.0 | Reject entries if current spread > threshold. |
| SPREAD_LOOKBACK | 50 | Rolling window for median/outlier detection. |

### 6.7 Trailing Stop (Core R‑Logic)
| Variable | Default | Effect |
|----------|---------|--------|
| TRAILING_ENABLED | true/false | Master switch for trailing behavior. |
| TRAILING_ACTIVATE_R | 1.0 | R at which initial move to (breakeven + buffer) triggers. |
| TRAILING_STEP_R | 0.5 | Additional R increments required to lock more. |
| TRAILING_MAX_LOCK_R | 5.0 | Cap on total locked R progression. |
| TRAILING_UPDATE_INTERVAL_SECONDS | 15 | Min seconds between stop modification attempts. |
| TRAILING_MIN_MOVE_PIPS | 3 | Minimum pip improvement required for a stop shift. |
| TRAILING_BREAKEVEN_BUFFER_PIPS | 1 | Buffer beyond entry when first moving stop. |

### 6.8 Trailing Mode Selection (R vs ATR)
| Variable | Default | Effect |
|----------|---------|--------|
| TRAILING_MODE | R | R = discrete R‑based ladder; ATR = volatility adaptive trailing. |
| ATR_PERIOD | 14 | Lookback used for ATR trailing. |
| ATR_MULTIPLIER | 2.0 | Distance = ATR * multiplier for stop placement. |

### 6.9 Data / Heartbeat
| Variable | Default | Effect |
|----------|---------|--------|
| MAX_TICK_STALENESS_SECONDS | 30 | Pause new entries if last tick older than this. |

### 6.10 Optional / Advanced (Hooks)
| Variable | Purpose |
|----------|---------|
| LOSS_STREAK_CONTRACT_THRESHOLD | Loss streak length to trigger risk contraction. |
| RISK_CONTRACTION_FACTOR | Multiplies base risk when contraction active. |
| RISK_RESTORE_STEP | Increment used to restore risk after wins. |
| MAX_CONCURRENT_TRADES | Hard concurrency limit (if implemented). |
| COOLDOWN_SECONDS_AFTER_LOSS | Enforced idle period after a loss. |

---

## 7. Quick Start (Windows)

```powershell
git clone <your-repo-url>
cd forex_bot
python -m venv forex_bot_env
./forex_bot_env/Scripts/activate
pip install -r requirements.txt
copy .env.example .env   # then edit values
python bot.py
```

To run continuously use NSSM (see previous version) or any process manager / scheduler.

---

## 8. Operating Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Live Trading | Normal (DRY_RUN=false) | Places real orders. |
| Dry Run | DRY_RUN=true | Logs decisions; no orders. |
| Fixed Size | FIXED_POSITION_SIZE set | Uses exact units. Risk % ignored. |
| Adaptive Virtual Equity | VIRTUAL_EQUITY_MODE=adaptive | Gradually increases effective equity under rules. |
| Long‑Only | LONG_ONLY=true | Filters out short signals. |
| ATR Trailing | TRAILING_ENABLED=true & TRAILING_MODE=ATR | Volatility reactive exit management. |

---

## 9. Journaling & Metrics

CSV columns include: timestamp, instrument, direction, entry, size, stop, tp, realized_r, unrealized_r (as updated), regime snapshot, volatility tags, phase events. Use:

```powershell
python journal_summary.py
```

Outputs win rate, expectancy, profit factor, regime distribution. Extend easily for equity curve & drawdown analytics.

---

## 10. Backtesting

`backtest_advanced_strategy.py` simulates bracket SL/TP and trailing rules for historical price series. Provide a CSV of prices, configure filename inside script, then run:

```powershell
python backtest_advanced_strategy.py
```

Generates `backtest_trades.csv` aligned with live journal schema for comparative analysis.

---

## 11. Safety & Risk Guidance

1. Always start DRY_RUN=true to validate logic & notifications.
2. Use VIRTUAL_EQUITY to tame oversized demo balances (avoid deceptive performance scaling).
3. Set MAX_UNITS_CAP conservatively while iterating.
4. Keep MIN_STOP_PIPS_OVERRIDE reasonable (too low = oversizing risk, too high = poor R:R).
5. DAILY_MAX_LOSS_R should reflect emotional capital tolerance; -3R to -5R common.
6. DAILY_MAX_PROFIT_R helps lock “good days” and prevent overtrading / reversal chop.
7. Heartbeat + spread filters reduce execution during outages / illiquid spikes.
8. Regularly archive / compress the journal (log growth management).
9. Reconcile open trades if manual platform intervention occurs.

---

## 12. Troubleshooting

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| No trades | Session guard / spread / staleness filter blocking | Check log tags: [SessionRiskGuard], [SpreadFilter], [Heartbeat] |
| Huge position size | Tight stop or low pip value estimate | Raise MIN_STOP_PIPS_OVERRIDE; verify instrument pip logic |
| Trailing not updating | TRAILING_ENABLED false or R threshold not reached | Check TRAILING_ACTIVATE_R & log messages |
| Telegram silent | Token/chat incorrect | Test with `python send_telegram.py` |
| Risk not matching expectation | FIXED_POSITION_SIZE active | Remove fixed size or adjust risk % |
| Adaptive not growing | Mode not set to adaptive or cooldown active | Check timestamps & ADAPT_COOLDOWN_SECONDS |

---

## 13. FAQ

**Q: Does fixed position size ignore risk %?**  Yes, risk % only applies when fixed size unset.

**Q: Which wins if both virtual equity and fixed size are set?**  Fixed size. Virtual equity remains dormant until fixed size removed.

**Q: Can I run multiple instruments?**  Extend the streaming client & loop to iterate instruments; ensure concurrency control & sizing isolation.

**Q: How accurate is pip value?**  Heuristic for major pairs; for cross‑currency enhancements add real FX conversion queries.

---

## 14. Roadmap / Extensibility Ideas

| Idea | Benefit |
|------|---------|
| Equity curve & drawdown analytics | Performance transparency |
| Advanced news blackout windows | Event risk avoidance |
| Multi‑instrument orchestration | Diversification |
| Regime‑adaptive TP multiples | Dynamic expectancy optimization |
| Machine learning feature scoring | Strategy evolution |

---

## 15. Security & Secrets

Do NOT commit real `.env` values. Use `.env.example` as template. Rotate API keys periodically. Logs may contain partial parameter values—sanitize before sharing externally.

---

## 16. License / Disclaimer

Educational purposes only. Trading involves substantial risk. No warranty expressed or implied.

---

## 17. Quick Reference Cheat Sheet

| Action | Command (PowerShell) |
|--------|----------------------|
| Activate env | `./forex_bot_env/Scripts/activate` |
| Run bot | `python bot.py` |
| Backtest | `python backtest_advanced_strategy.py` |
| Journal summary | `python journal_summary.py` |
| Test Telegram | `python send_telegram.py` |

---

Happy building & safe trading.

