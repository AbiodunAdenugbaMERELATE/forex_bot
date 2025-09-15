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

    def get_account_summary(self):
        """Return basic account summary including balance used for position sizing."""
        url = f'{self.base_url}/accounts/{self.account_id}/summary'
        r = requests.get(url, headers=self.headers, timeout=10)
        data = r.json()
        # Normalize to expected structure
        if 'account' in data:
            account = data['account']
            return {
                'balance': account.get('balance'),
                'currency': account.get('currency'),
                'unrealizedPL': account.get('unrealizedPL'),
                'NAV': account.get('NAV')
            }
        return {'balance': 0}

    def get_open_trades_count(self, instrument=None):
        """Return number of open trades (optionally filtered by instrument)."""
        url = f'{self.base_url}/accounts/{self.account_id}/openTrades'
        r = requests.get(url, headers=self.headers, timeout=10)
        data = r.json()
        trades = data.get('trades', [])
        if instrument:
            trades = [t for t in trades if t.get('instrument') == instrument]
        return len(trades)
