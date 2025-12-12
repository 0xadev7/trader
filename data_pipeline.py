"""Data collection and preprocessing pipeline."""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import ta
from loguru import logger
from config import Config
from gateio_client import GateIOClient, TradingPair


class DataPipeline:
    """Handles data collection, cleaning, and feature engineering."""
    
    def __init__(self, client: GateIOClient):
        self.client = client
    
    def fetch_historical_data(self, pair: str, interval: str, 
                            limit: int = 500) -> pd.DataFrame:
        """Fetch historical candlestick data."""
        try:
            pair_obj = TradingPair(pair, self.client)
            df = pair_obj.get_klines_df(interval, limit)
            logger.info(f"Fetched {len(df)} candles for {pair} ({interval})")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_14'] = df['rsi']
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close'], window=14
        ).adx()
        
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Momentum and ROC
        df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
        df['roc'] = df['momentum']
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # High-Low spread
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str],
                        target_col: str, sequence_length: int, 
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/Transformer models.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name (e.g., 'returns')
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
        
        Returns:
            X: Input sequences (samples, timesteps, features)
            y: Target values (samples,)
        """
        if df.empty:
            return np.array([]), np.array([])
        
        # Select features
        feature_data = df[feature_cols].values
        target_data = df[target_col].values
        
        # Remove NaN values
        valid_indices = ~(np.isnan(feature_data).any(axis=1) | 
                         np.isnan(target_data))
        feature_data = feature_data[valid_indices]
        target_data = target_data[valid_indices]
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - sequence_length - prediction_horizon + 1):
            X.append(feature_data[i:i+sequence_length])
            y.append(target_data[i+sequence_length+prediction_horizon-1])
        
        return np.array(X), np.array(y)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set for model training."""
        df = self.calculate_technical_indicators(df)
        
        # Drop rows with NaN values (from indicator calculations)
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training."""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        return base_features + Config.TECHNICAL_INDICATORS
    
    def calculate_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Calculate target variable (future returns)."""
        future_close = df['close'].shift(-horizon)
        returns = (future_close - df['close']) / df['close']
        return returns

