"""Example usage of the trading bot components."""
from config import Config
from gateio_client import GateIOClient, TradingPair
from data_pipeline import DataPipeline
from loguru import logger

# Example 1: Fetch and analyze market data
def example_fetch_data():
    """Example of fetching and analyzing market data."""
    logger.info("Example 1: Fetching market data")
    
    # Initialize client (you'll need API credentials)
    client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)
    
    # Initialize data pipeline
    pipeline = DataPipeline(client)
    
    # Fetch historical data
    df = pipeline.fetch_historical_data('BTC_USDT', '15m', limit=500)
    
    if not df.empty:
        # Calculate technical indicators
        df = pipeline.prepare_features(df)
        
        logger.info(f"Fetched {len(df)} candles")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Latest close: ${df.iloc[-1]['close']:.2f}")
        logger.info(f"RSI: {df.iloc[-1]['rsi']:.2f}")
        logger.info(f"MACD: {df.iloc[-1]['macd']:.2f}")


# Example 2: Get current price
def example_get_price():
    """Example of getting current price."""
    logger.info("Example 2: Getting current price")
    
    client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)
    pair = TradingPair('BTC_USDT', client)
    
    price = pair.get_current_price()
    logger.info(f"Current BTC price: ${price:.2f}")


# Example 3: Check account balance
def example_check_balance():
    """Example of checking account balance."""
    logger.info("Example 3: Checking account balance")
    
    client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)
    
    try:
        balances = client.get_account_balance()
        for balance in balances:
            currency = balance['currency']
            available = balance.get('available', '0')
            locked = balance.get('locked', '0')
            
            if float(available) > 0 or float(locked) > 0:
                logger.info(f"{currency}: Available={available}, Locked={locked}")
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")


if __name__ == '__main__':
    logger.info("Trading Bot - Example Usage")
    logger.info("=" * 50)
    
    # Uncomment examples to run:
    # example_fetch_data()
    # example_get_price()
    # example_check_balance()
    
    logger.info("\nNote: Set GATE_API_KEY and GATE_API_SECRET in .env file")
    logger.info("Uncomment examples in the script to run them")

