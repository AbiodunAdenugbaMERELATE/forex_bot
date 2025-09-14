import requests
import os
from dotenv import load_dotenv

load_dotenv()

class OandaAPI:
    def __init__(self):
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.base_url = 'https://api-fxpractice.oanda.com/v3'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def place_order(self, instrument, units, order_type='MARKET'):
        url = f'{self.base_url}/accounts/{self.account_id}/orders'
        order_data = {
            "order": {
                "units": str(units),
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": order_type,
                "positionFill": "DEFAULT"
            }
        }
        response = requests.post(url, headers=self.headers, json=order_data)
        return response.json()
