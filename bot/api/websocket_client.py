import json
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream
import threading
import os
from dotenv import load_dotenv
import oandapyV20.exceptions

load_dotenv()

ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
ACCESS_TOKEN = os.getenv('OANDA_API_KEY')

api = API(access_token=ACCESS_TOKEN, environment="practice")

def on_message(data, strategies):
    try:
        if 'type' in data and data['type'] == 'PRICE':
            instrument = data.get('instrument')
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            bid = float(bids[0]['price']) if bids else None
            ask = float(asks[0]['price']) if asks else None
            if bid is not None and ask is not None:
                data['spread'] = ask - bid
                data['mid'] = (ask + bid) / 2.0
            if instrument in strategies:
                strategies[instrument].on_price_update(data)
    except Exception as e:
        print(f"on_message processing error: {e}")

def stream_data(instruments, strategies):
    params = {"instruments": ",".join(instruments)}
    s = PricingStream(accountID=ACCOUNT_ID, params=params)
    
    try:
        for R in api.request(s):
            on_message(R, strategies)
    except oandapyV20.exceptions.V20Error as e:
        print(f"Error: {e}")

def start_stream_thread(instruments, strategies):
    stream_thread = threading.Thread(target=stream_data, args=(instruments, strategies))
    stream_thread.start()
    return stream_thread
