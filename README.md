# Forex Trading Bot

A modular, event-driven Forex trading bot for OANDA, featuring advanced risk management, live Telegram notifications, and automated strategy optimization.

---

## Features

- Connects to OANDA for live Forex trading via API and websocket.
- Implements advanced trading strategies with risk controls and market regime detection.
- Sends all trade events and errors to Telegram.
- Supports backtesting with historical data.
- Designed for continuous, unattended operation.

---

## Setup Instructions (Windows)

### 1. Clone the Repository

```powershell
git clone <your-repo-url>
cd forex_bot
```

### 2. Set Up Python Environment

Install Python 3.11+ from [python.org](https://www.python.org/downloads/).

Create and activate a virtual environment:

```powershell
python -m venv forex_bot_env
.\forex_bot_env\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your credentials:

```
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
FMP_API_KEY=your_financial_modeling_prep_api_key
\n+# --- Position Sizing & Virtual Equity Controls (optional) ---
# Treat the account as if it only has this much equity (helps when demo balance is larger)
VIRTUAL_EQUITY=2000
# scale | cap | adaptive
VIRTUAL_EQUITY_MODE=scale
# Risk per trade (override dynamic adjustments) e.g. 0.005 = 0.5%
MAX_RISK_PER_TRADE=0.005
# Absolute ceiling on units sent in any single order
MAX_UNITS_CAP=5000
# Do not use more than this fraction of available margin for a new position
MARGIN_USAGE_CAP_PCT=0.20
# Minimum enforced stop in pips (prevents tiny stops creating huge sizes)
MIN_STOP_PIPS_OVERRIDE=15
# Account base currency (assumed for pip value simplification)
ACCOUNT_CURRENCY=USD
# adaptive mode parameters (only used if VIRTUAL_EQUITY_MODE=adaptive)
ADAPT_PARTICIPATION_PCT=0.25
ADAPT_MAX_MULTIPLIER=1.5
ADAPT_COOLDOWN_SECONDS=3600
# Dry-run mode: log signals & sizing but do not place real orders
DRY_RUN=false
```

---

## What the Bot Does

- Listens to live price data via websocket.
- Analyzes market conditions and risk.
- Places trades automatically when strategy conditions are met.
- Sends all trade events (buy, sell, errors, warnings) to your Telegram.
- Logs all activity to `forex_bot.log`.
- Dynamically sizes positions using: effective_balance * risk% / (stop_loss_pips * pip_value),
  then volatility adjustment and safety caps (virtual equity + max units + margin usage cap).

---

## Running the Bot (Windows)

### **Manual Start (in a terminal window)**

1. Activate your environment:
   ```powershell
   .\forex_bot_env\Scripts\activate
   ```
2. Start the bot:
   ```powershell
   python bot.py
   ```
   - Keep this window open for the bot to keep running.

---

### **Run the Bot as a Windows Service (Recommended for 24/7 operation)**

1. Download [NSSM](https://nssm.cc/download) and extract it.
2. Open Command Prompt as Administrator.
3. Install the bot as a service:
   ```powershell
   nssm install ForexBotService
   ```
   - For "Application", browse to:  
     `C:\Users\<your-user>\Desktop\forex_bot\forex_bot_env\Scripts\python.exe`
   - For "Arguments", enter:  
     `C:\Users\<your-user>\Desktop\forex_bot\bot.py`
   - For "Startup directory", use:  
     `C:\Users\<your-user>\Desktop\forex_bot`
4. Start the service:
   ```powershell
   nssm start ForexBotService
   ```
5. The bot will now run in the background and restart automatically if your computer reboots.

---

## Backtesting

To backtest your strategy:

1. Place a historical CSV file (e.g., `EURUSD_3years.csv`) in the project folder with columns: `timestamp,mid`.
2. Edit `backtest_advanced_strategy.py` to set the filename.
3. Run:
   ```powershell
   python backtest_advanced_strategy.py
   ```

---

## Troubleshooting

- Check `forex_bot.log` for errors.
- Ensure all API keys and credentials are correct in your `.env` file.
- For Telegram issues, test with `python send_telegram.py`.
- If position sizes look too large:
   - Reduce `MAX_RISK_PER_TRADE`.
   - Lower `MAX_UNITS_CAP`.
   - Set `VIRTUAL_EQUITY` and `VIRTUAL_EQUITY_MODE=scale`.
   - Increase `MIN_STOP_PIPS_OVERRIDE` if stops too tight.
   - Check logs for `[PositionSizing]` lines to see raw vs final size.

---

## Notes

- The bot will not trade on weekends (when Forex markets are closed).
- Always test with a demo account before trading live.
- Monitor your bot and logs regularly.

---
