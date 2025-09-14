import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'bot')))

import threading
import signal
import time
import logging
from bot.api.oanda_api import OandaAPI
from bot.strategies.advanced_strategy import AdvancedStrategy
from bot.api.websocket_client import start_stream_thread
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received")
    raise SystemExit

def start_bot():
    # Load environment variables
    load_dotenv()

    signal.signal(signal.SIGINT, handle_shutdown)
    api_client = OandaAPI()
    logger.info("API client initialized")

    # Major and minor currency pairs
    symbols = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", 
        "USD_CAD", "USD_CHF", "NZD_USD", "EUR_GBP", 
        "EUR_JPY", "GBP_JPY"
    ]
    
    strategies = {}

    for symbol in symbols:
        strategies[symbol] = AdvancedStrategy(api_client, symbol, strategies)
        logger.info(f"Advanced strategy initialized for {symbol}")

    logger.info("Starting data stream...")
    stream_thread = start_stream_thread(symbols, strategies)
    logger.info("Data stream started")

    try:
        logger.info("Entering main loop")
        while True:
            if not stream_thread.is_alive():
                logger.error("Stream thread is no longer alive. Restarting the bot...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping...")
    finally:
        logger.info("Stopping stream thread...")
        stream_thread.join()
        logger.info("Stream thread stopped")

if __name__ == "__main__":
    while True:
        try:
            start_bot()
        except Exception as e:
            logger.error(f"Bot terminated unexpectedly. Error: {e}")
        
        logger.info("Restarting the bot in 5 seconds...")
        time.sleep(5)
