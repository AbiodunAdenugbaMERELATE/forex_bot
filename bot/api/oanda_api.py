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

    def place_order(self, instrument, units, order_type='MARKET',
                    stop_loss_price=None, take_profit_price=None, trailing_stop_distance=None):
        """Place an order with optional bracket components.

        Parameters:
            instrument (str): e.g. 'EUR_USD'
            units (int): positive for buy (long), negative for sell (short)
            order_type (str): 'MARKET' only currently
            stop_loss_price (float|None): absolute price for stop loss
            take_profit_price (float|None): absolute price for take profit
            trailing_stop_distance (float|None): distance in ABSOLUTE price units (not pips)
        """
        url = f'{self.base_url}/accounts/{self.account_id}/orders'
        order = {
            "units": str(int(units)),
            "instrument": instrument,
            "timeInForce": "FOK" if order_type == 'MARKET' else "GTC",
            "type": order_type,
            "positionFill": "DEFAULT"
        }
        if stop_loss_price is not None:
            try:
                order["stopLossOnFill"] = {"price": f"{float(stop_loss_price):.5f}"}
            except Exception:
                pass
        if take_profit_price is not None:
            try:
                order["takeProfitOnFill"] = {"price": f"{float(take_profit_price):.5f}"}
            except Exception:
                pass
        if trailing_stop_distance is not None:
            try:
                order["trailingStopLossOnFill"] = {"distance": f"{float(trailing_stop_distance):.5f}"}
            except Exception:
                pass
        payload = {"order": order}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            return response.json()
        except Exception as e:
            return {"error": str(e), "payload": payload}

    def modify_trade_stops(self, trade_id, stop_loss_price=None, take_profit_price=None, trailing_distance=None):
        """Modify existing trade protective orders.

        OANDA requires separate order objects for modifying stops/TP. This method attempts
        to submit the relevant amendments. If nothing provided, returns early.
        """
        if not any([stop_loss_price, take_profit_price, trailing_distance]):
            return {"error": "No modifications specified"}
        url = f'{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/orders'
        orders = []
        if stop_loss_price is not None:
            try:
                orders.append({
                    "type": "STOP_LOSS",
                    "tradeID": str(trade_id),
                    "price": f"{float(stop_loss_price):.5f}",
                    "timeInForce": "GTC"
                })
            except Exception:
                pass
        if take_profit_price is not None:
            try:
                orders.append({
                    "type": "TAKE_PROFIT",
                    "tradeID": str(trade_id),
                    "price": f"{float(take_profit_price):.5f}",
                    "timeInForce": "GTC"
                })
            except Exception:
                pass
        if trailing_distance is not None:
            try:
                orders.append({
                    "type": "TRAILING_STOP_LOSS",
                    "tradeID": str(trade_id),
                    "distance": f"{float(trailing_distance):.5f}",
                    "timeInForce": "GTC"
                })
            except Exception:
                pass
        if not orders:
            return {"error": "No valid orders built"}
        payload = {"orders": orders}
        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=15)
            return r.json()
        except Exception as e:
            return {"error": str(e), "payload": payload}

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
