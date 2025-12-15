"""Configuration settings for the trading bot."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gate.io API
    GATE_API_KEY = os.getenv('GATE_API_KEY', '')
    GATE_API_SECRET = os.getenv('GATE_API_SECRET', '')
    GATE_API_URL = 'https://api.gateio.ws/api/v4'
    
    # Trading Configuration
    BASE_CURRENCY = os.getenv('BASE_CURRENCY', 'USDT')
    TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTC_USDT,ETH_USDT').split(',')
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% risk per trade
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% max position
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
    DATA_PATH = os.getenv('DATA_PATH', './data')
    LOG_PATH = os.getenv('LOG_PATH', './logs')
    
    # Trading Parameters
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    PRIMARY_TIMEFRAME = os.getenv('PRIMARY_TIMEFRAME', '15m')
    LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', '100'))
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '5'))
    
    # AI Model Parameters
    LSTM_UNITS = int(os.getenv('LSTM_UNITS', '128'))
    TRANSFORMER_D_MODEL = int(os.getenv('TRANSFORMER_D_MODEL', '128'))
    TRANSFORMER_NHEAD = int(os.getenv('TRANSFORMER_NHEAD', '8'))
    TRANSFORMER_NLAYERS = int(os.getenv('TRANSFORMER_NLAYERS', '4'))
    DROPOUT_RATE = float(os.getenv('DROPOUT_RATE', '0.2'))
    
    # Reinforcement Learning
    RL_GAMMA = float(os.getenv('RL_GAMMA', '0.99'))
    RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', '3e-4'))
    RL_BATCH_SIZE = int(os.getenv('RL_BATCH_SIZE', '64'))
    RL_BUFFER_SIZE = int(os.getenv('RL_BUFFER_SIZE', '10000'))
    
    # Risk Management
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))  # 2% stop loss
    TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.04'))  # 4% take profit
    MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT', '0.15'))  # 15% max drawdown
    
    # Training data date range
    TRAIN_START_DATE = os.getenv('TRAIN_START_DATE', '2023-01-01')
    TRAIN_END_DATE = os.getenv('TRAIN_END_DATE', '2024-01-01')
    
    # Backtesting
    BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2024-01-01')
    BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2024-12-31')
    BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '10000'))
    BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.001'))  # 0.1% commission
    
    # Dry Run Configuration
    DRY_RUN_DURATION_DAYS = int(os.getenv('DRY_RUN_DURATION_DAYS', '7'))
    DRY_RUN_INTERVAL_MINUTES = int(os.getenv('DRY_RUN_INTERVAL_MINUTES', '5'))
    
    # Trading Bot Configuration
    TRADING_INTERVAL_MINUTES = int(os.getenv('TRADING_INTERVAL_MINUTES', '5'))
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26',
        'rsi', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr', 'adx',
        'obv', 'volume_sma',
        'momentum', 'roc'
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.MODEL_PATH, exist_ok=True)
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        os.makedirs(cls.LOG_PATH, exist_ok=True)

