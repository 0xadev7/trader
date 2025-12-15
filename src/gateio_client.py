"""Gate.io API client for trading operations."""

import time
import hmac
import hashlib
import base64
import requests
from typing import Dict, List
from loguru import logger

from .config import Config


class GateIOClient:
    """Client for interacting with Gate.io API."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = Config.GATE_API_URL
        self.session = requests.Session()

    def _sign(
        self, method: str, url: str, query_string: str = "", payload_string: str = ""
    ) -> Dict[str, str]:
        """Generate signature for API authentication."""
        t = str(int(time.time()))
        m = hashlib.sha512()
        m.update((payload_string or "").encode("utf-8"))
        hashed_payload = m.hexdigest()
        s = f"{method}\n{url}\n{query_string}\n{hashed_payload}\n{t}"
        sign = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"), s.encode("utf-8"), hashlib.sha512
            ).digest()
        ).decode("utf-8")

        return {"KEY": self.api_key, "Timestamp": t, "SIGN": sign}

    def _request(
        self, method: str, endpoint: str, params: Dict = None, data: Dict = None
    ) -> Dict:
        """Make authenticated API request."""
        url = f"{self.base_url}/{endpoint}"
        query_string = ""

        if params:
            # URL encode parameter values
            from urllib.parse import urlencode
            query_string = urlencode(sorted(params.items()))

        payload_string = ""
        if data:
            import json

            payload_string = json.dumps(data)

        headers = self._sign(
            method, f"/api/v4/{endpoint}", query_string, payload_string
        )
        headers["Content-Type"] = "application/json"

        try:
            if method == "GET":
                response = self.session.get(
                    url, headers=headers, params=params, timeout=10
                )
            elif method == "POST":
                response = self.session.post(
                    url, headers=headers, params=params, json=data, timeout=10
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, headers=headers, params=params, timeout=10
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e.response, 'text'):
                try:
                    error_detail = e.response.json()
                    error_msg += f" - Details: {error_detail}"
                except:
                    error_msg += f" - Response: {e.response.text[:200]}"
            logger.error(error_msg)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_account_balance(self) -> Dict:
        """Get account balance."""
        return self._request("GET", "spot/accounts")

    def get_ticker(self, pair: str) -> Dict:
        """Get ticker information for a trading pair."""
        return self._request("GET", f"spot/tickers", {"currency_pair": pair})

    def get_orderbook(self, pair: str, limit: int = 20) -> Dict:
        """Get orderbook for a trading pair."""
        return self._request(
            "GET", f"spot/order_book", {"currency_pair": pair, "limit": limit}
        )

    def get_klines(
        self, pair: str, interval: str = "15m", limit: int = 100, from_time: int = None, to_time: int = None
    ) -> List:
        """Get candlestick data (klines).

        Args:
            pair: Trading pair (e.g., 'BTC_USDT')
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000 per request)
            from_time: Start timestamp in seconds (optional)
            to_time: End timestamp in seconds (optional)
        """
        # Gate.io API v4 uses BTC_USDT format (underscore) for currency_pair
        # Limit maximum is typically 1000 per request
        max_limit = min(limit, 1000)
        params = {"currency_pair": pair, "interval": interval, "limit": max_limit}
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time

        return self._request("GET", "spot/candlesticks", params)

    def place_order(
        self,
        pair: str,
        side: str,
        amount: float,
        price: float = None,
        order_type: str = "limit",
    ) -> Dict:
        """Place an order.

        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (required for limit orders)
            order_type: 'limit' or 'market'
        """
        data = {
            "currency_pair": pair,
            "side": side,
            "amount": str(amount),
            "type": order_type,
        }

        if order_type == "limit" and price:
            data["price"] = str(price)

        return self._request("POST", "spot/orders", data=data)

    def cancel_order(self, pair: str, order_id: str) -> Dict:
        """Cancel an order."""
        return self._request(
            "DELETE", f"spot/orders/{order_id}", params={"currency_pair": pair}
        )

    def get_open_orders(self, pair: str = None) -> List:
        """Get open orders."""
        params = {}
        if pair:
            params["currency_pair"] = pair
        return self._request("GET", "spot/open_orders", params)

    def get_order_status(self, pair: str, order_id: str) -> Dict:
        """Get order status."""
        return self._request(
            "GET", f"spot/orders/{order_id}", params={"currency_pair": pair}
        )

    def get_my_trades(self, pair: str = None, limit: int = 100) -> List:
        """Get trade history."""
        params = {"limit": limit}
        if pair:
            params["currency_pair"] = pair
        return self._request("GET", "spot/my_trades", params)

    def convert_pair_format(self, pair: str) -> str:
        """Convert pair format (BTC_USDT -> BTC/USDT)."""
        return pair.replace("_", "/")


class TradingPair:
    """Represents a trading pair with market data."""

    def __init__(self, symbol: str, client: GateIOClient):
        self.symbol = symbol
        self.client = client
        self.base_asset, self.quote_asset = symbol.split("_")

    def get_current_price(self) -> float:
        """Get current market price."""
        ticker = self.client.get_ticker(self.symbol)
        return float(ticker.get("last", 0))

    def get_klines_df(self, interval: str, limit: int = None, start_date: str = None, end_date: str = None) -> "pd.DataFrame":
        """Get candlestick data as DataFrame.
        
        Args:
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (optional, ignored if date range provided)
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
        
        If date range is provided, fetches all candles in that range.
        Otherwise uses limit to fetch recent candles.
        """
        import pandas as pd
        import time as time_module
        from datetime import datetime
        
        all_klines = []
        
        # Convert date strings to timestamps if provided
        from_time = None
        to_time = None
        if start_date:
            from_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        if end_date:
            # Set to end of day
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            to_time = int(end_dt.replace(hour=23, minute=59, second=59).timestamp())
        
        # Gate.io API max limit per request is 1000
        max_per_request = 1000
        
        # Calculate interval duration in seconds for date range fetching
        interval_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }.get(interval, 900)
        
        if from_time and to_time:
            # Fetch by date range - work backwards from end_date
            # Gate.io API has a limit on the time range per request (max 1000 candles worth)
            # So we need to fetch in smaller time windows
            current_to = to_time
            estimated_candles = (to_time - from_time) // interval_seconds
            logger.info(f"Fetching data from {start_date} to {end_date} (estimated {estimated_candles} candles)")
            
            # Calculate max time range for 1000 candles (use 900 to be safe)
            max_candles_per_request = 900
            max_time_range = max_candles_per_request * interval_seconds
            
            while current_to > from_time:
                try:
                    # Calculate the from_time for this request (max 900 candles back)
                    request_from = max(from_time, current_to - max_time_range)
                    
                    klines = self.client.get_klines(
                        self.symbol, interval, max_candles_per_request, 
                        from_time=request_from, to_time=current_to
                    )
                    
                    if not klines:
                        break
                    
                    # Gate.io returns data in reverse chronological order (newest first)
                    # Prepend to maintain chronological order when we reverse later
                    all_klines = klines + all_klines
                    
                    # Get the oldest timestamp from this batch (last item in response)
                    oldest_timestamp = int(klines[-1][0])
                    
                    # If we've reached or passed the start time, we're done
                    if oldest_timestamp <= from_time:
                        # Filter out any candles before start_time
                        all_klines = [k for k in all_klines if int(k[0]) >= from_time]
                        break
                    
                    # Move the window backwards (use oldest timestamp - 1 to avoid overlap)
                    current_to = oldest_timestamp - 1
                    
                    # Rate limiting
                    time_module.sleep(0.15)
                    
                    # Progress logging
                    if len(all_klines) % 5000 == 0:
                        oldest_dt = pd.to_datetime(oldest_timestamp, unit='s')
                        logger.info(f"Fetched {len(all_klines)} candles so far... (oldest: {oldest_dt})")
                        
                except Exception as e:
                    logger.error(f"Error fetching klines batch: {e}")
                    # Try with smaller range if range too broad error
                    if "range too broad" in str(e).lower() or "INVALID_PARAM_VALUE" in str(e):
                        logger.warning("Range too broad, trying with smaller time window...")
                        # Reduce max candles and try again
                        max_candles_per_request = min(max_candles_per_request - 100, 500)
                        max_time_range = max_candles_per_request * interval_seconds
                        if max_candles_per_request < 100:
                            logger.error("Cannot fetch data - range is too broad even with minimum window")
                            break
                        continue
                    import traceback
                    traceback.print_exc()
                    break
        else:
            # Fetch by limit (backward compatibility)
            remaining_limit = limit or 500
            from_time_param = None
            
            while remaining_limit > 0:
                request_limit = min(remaining_limit, max_per_request)
                try:
                    klines = self.client.get_klines(
                        self.symbol, interval, request_limit, from_time=from_time_param
                    )
                    
                    if not klines:
                        break
                        
                    all_klines.extend(klines)
                    
                    # If we got fewer than requested, we've reached the end
                    if len(klines) < request_limit:
                        break
                    
                    # Update from_time to get older data (klines are in reverse chronological order)
                    if len(klines) > 0:
                        from_time_param = int(klines[-1][0]) - 1
                    
                    remaining_limit -= len(klines)
                    
                    # Rate limiting
                    if remaining_limit > 0:
                        time_module.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error fetching klines batch: {e}")
                    break
        
        if not all_klines:
            return pd.DataFrame()

        # Gate.io API v4 candlesticks response format:
        # [t, v, c, h, l, o, sum, ...]
        # t: timestamp (Unix seconds)
        # v: volume (quote currency, e.g., USDT)
        # c: close price
        # h: high price
        # l: low price
        # o: open price
        # sum: volume (base currency, e.g., BTC)
        # Additional fields may be present
        
        # Check the actual number of columns in the response
        num_cols = len(all_klines[0]) if all_klines else 0
        
        if num_cols == 0:
            logger.warning("No klines data received")
            return pd.DataFrame()
        
        # Gate.io API v4 candlesticks format: [t, v, c, h, l, o, sum, ...]
        # Where: t=timestamp, v=quote_volume, c=close, h=high, l=low, o=open, sum=base_volume
        # We need at least 6 columns, but API may return 8 or more
        if num_cols < 6:
            logger.error(f"Unexpected klines format: {num_cols} columns, expected at least 6")
            logger.debug(f"First kline sample: {all_klines[0] if all_klines else 'None'}")
            return pd.DataFrame()
        
        # Create DataFrame - let pandas handle the column count automatically
        # Then select and rename only the columns we need
        df = pd.DataFrame(all_klines)
        
        # Gate.io API returns columns in order: [timestamp, quote_volume, close, high, low, open, base_volume, ...]
        # Map to our standard format: [timestamp, open, high, low, close, volume]
        if num_cols >= 6:
            df = df.iloc[:, [0, 5, 3, 4, 2, 1]].copy()  # Select columns: [timestamp, open, high, low, close, volume]
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        else:
            logger.error(f"Insufficient columns in klines response: {num_cols}")
            return pd.DataFrame()

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Sort by timestamp (oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Remove duplicates (in case of overlap)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include entire end date
            df = df[df['timestamp'] < end_dt]
        
        # Limit to requested amount if limit was specified and no date range
        if limit and not (start_date or end_date):
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
        
        return df
