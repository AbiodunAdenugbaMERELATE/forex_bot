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
    print(f"Received message: {json.dumps(data, indent=2)}")
    if 'type' in data and data['type'] == 'PRICE':
        instrument = data['instrument']
        if instrument in strategies:
            strategies[instrument].on_price_update(data)

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
