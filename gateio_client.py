"""Gate.io API client for trading operations."""
import time
import hmac
import hashlib
import base64
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
from config import Config


class GateIOClient:
    """Client for interacting with Gate.io API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = Config.GATE_API_URL
        self.session = requests.Session()
        
    def _sign(self, method: str, url: str, query_string: str = '', 
              payload_string: str = '') -> Dict[str, str]:
        """Generate signature for API authentication."""
        t = str(int(time.time()))
        m = hashlib.sha512()
        m.update((payload_string or '').encode('utf-8'))
        hashed_payload = m.hexdigest()
        s = f'{method}\n{url}\n{query_string}\n{hashed_payload}\n{t}'
        sign = base64.b64encode(
            hmac.new(self.api_secret.encode('utf-8'),
                    s.encode('utf-8'),
                    hashlib.sha512).digest()
        ).decode('utf-8')
        
        return {
            'KEY': self.api_key,
            'Timestamp': t,
            'SIGN': sign
        }
    
    def _request(self, method: str, endpoint: str, params: Dict = None, 
                 data: Dict = None) -> Dict:
        """Make authenticated API request."""
        url = f"{self.base_url}/{endpoint}"
        query_string = ''
        
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        payload_string = ''
        if data:
            import json
            payload_string = json.dumps(data)
        
        headers = self._sign(method, f"/api/v4/{endpoint}", query_string, payload_string)
        headers['Content-Type'] = 'application/json'
        
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, params=params, 
                                            json=data, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_account_balance(self) -> Dict:
        """Get account balance."""
        return self._request('GET', 'spot/accounts')
    
    def get_ticker(self, pair: str) -> Dict:
        """Get ticker information for a trading pair."""
        return self._request('GET', f'spot/tickers', {'currency_pair': pair})
    
    def get_orderbook(self, pair: str, limit: int = 20) -> Dict:
        """Get orderbook for a trading pair."""
        return self._request('GET', f'spot/order_book', 
                           {'currency_pair': pair, 'limit': limit})
    
    def get_klines(self, pair: str, interval: str = '15m', 
                   limit: int = 100, from_time: int = None) -> List:
        """Get candlestick data (klines).
        
        Args:
            pair: Trading pair (e.g., 'BTC_USDT')
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles
            from_time: Start timestamp (optional)
        """
        params = {
            'currency_pair': pair,
            'interval': interval,
            'limit': limit
        }
        if from_time:
            params['from'] = from_time
        
        return self._request('GET', 'spot/candlesticks', params)
    
    def place_order(self, pair: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'limit') -> Dict:
        """Place an order.
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (required for limit orders)
            order_type: 'limit' or 'market'
        """
        data = {
            'currency_pair': pair,
            'side': side,
            'amount': str(amount),
            'type': order_type
        }
        
        if order_type == 'limit' and price:
            data['price'] = str(price)
        
        return self._request('POST', 'spot/orders', data=data)
    
    def cancel_order(self, pair: str, order_id: str) -> Dict:
        """Cancel an order."""
        return self._request('DELETE', f'spot/orders/{order_id}',
                           params={'currency_pair': pair})
    
    def get_open_orders(self, pair: str = None) -> List:
        """Get open orders."""
        params = {}
        if pair:
            params['currency_pair'] = pair
        return self._request('GET', 'spot/open_orders', params)
    
    def get_order_status(self, pair: str, order_id: str) -> Dict:
        """Get order status."""
        return self._request('GET', f'spot/orders/{order_id}',
                           params={'currency_pair': pair})
    
    def get_my_trades(self, pair: str = None, limit: int = 100) -> List:
        """Get trade history."""
        params = {'limit': limit}
        if pair:
            params['currency_pair'] = pair
        return self._request('GET', 'spot/my_trades', params)
    
    def convert_pair_format(self, pair: str) -> str:
        """Convert pair format (BTC_USDT -> BTC/USDT)."""
        return pair.replace('_', '/')


class TradingPair:
    """Represents a trading pair with market data."""
    
    def __init__(self, symbol: str, client: GateIOClient):
        self.symbol = symbol
        self.client = client
        self.base_asset, self.quote_asset = symbol.split('_')
        
    def get_current_price(self) -> float:
        """Get current market price."""
        ticker = self.client.get_ticker(self.symbol)
        return float(ticker.get('last', 0))
    
    def get_klines_df(self, interval: str, limit: int = 500) -> 'pd.DataFrame':
        """Get candlestick data as DataFrame."""
        import pandas as pd
        
        klines = self.client.get_klines(self.symbol, interval, limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'volume', 'close', 'high', 'low', 'open'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

