"""Backtesting framework for strategy validation."""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger

from .config import Config
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .ensemble_strategy import EnsembleStrategy
from .risk_manager import RiskManager
from .gateio_client import GateIOClient


class Backtester:
    """Comprehensive backtesting framework."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(self, data: pd.DataFrame, strategy: EnsembleStrategy,
                    risk_manager: RiskManager, pair: str) -> Dict:
        """Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data and features
            strategy: Ensemble strategy instance
            risk_manager: Risk manager instance
            pair: Trading pair symbol
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {pair} with {len(data)} candles")
        
        # Initialize tracking
        equity_curve = [self.initial_capital]
        trades = []
        current_position = None
        
        # Prepare sequences for prediction
        feature_cols = ['open', 'high', 'low', 'close', 'volume'] + \
                      [col for col in Config.TECHNICAL_INDICATORS if col in data.columns]
        
        sequence_length = Config.LOOKBACK_WINDOW
        prediction_horizon = Config.PREDICTION_HORIZON
        
        # Process each candle
        for i in range(sequence_length, len(data) - prediction_horizon):
            current_data = data.iloc[:i+1].copy()
            current_price = current_data.iloc[-1]['close']
            
            # Prepare features
            feature_data = current_data[feature_cols].values
            
            # Create sequences (use last sequence_length candles)
            if len(feature_data) >= sequence_length:
                X_seq = feature_data[-sequence_length:].reshape(1, sequence_length, -1)
                prices_seq = current_data['close'].values[-sequence_length:]
                
                # Get prediction from strategy
                try:
                    prediction = strategy.predict(X_seq, prices_seq[-1:])
                    signal = prediction['signal']
                    confidence = prediction['confidence']
                except Exception as e:
                    logger.error(f"Prediction error at step {i}: {e}")
                    signal = 'hold'
                    confidence = 0.0
            else:
                signal = 'hold'
                confidence = 0.0
            
            # Check existing position
            if pair in risk_manager.positions:
                position = risk_manager.positions[pair]
                entry_price = position['entry_price']
                
                # Check stop loss / take profit
                sl_tp = risk_manager.check_stop_loss_take_profit(pair, current_price)
                if sl_tp:
                    # Close position
                    close_result = risk_manager.close_position(pair, current_price)
                    trades.append({
                        'entry_time': current_data.iloc[-1]['timestamp'],
                        'exit_time': current_data.iloc[-1]['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'side': position['side'],
                        'pnl': close_result['pnl'],
                        'pnl_pct': close_result['pnl_pct'],
                        'exit_reason': sl_tp
                    })
                    current_position = None
            
            # Execute new trades based on signal
            if signal in ['buy', 'sell'] and confidence > 0.6:
                # Validate trade
                validation = risk_manager.validate_trade(
                    pair, signal, 
                    risk_manager.calculate_position_size(pair, current_price, confidence),
                    current_price
                )
                
                if validation['allowed']:
                    # Close opposite position if exists
                    if pair in risk_manager.positions:
                        position = risk_manager.positions[pair]
                        if position['side'] != signal:
                            close_result = risk_manager.close_position(pair, current_price)
                            trades.append({
                                'entry_time': current_data.iloc[-2]['timestamp'],
                                'exit_time': current_data.iloc[-1]['timestamp'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'side': position['side'],
                                'pnl': close_result['pnl'],
                                'pnl_pct': close_result['pnl_pct'],
                                'exit_reason': 'reverse'
                            })
                    
                    # Open new position
                    if pair not in risk_manager.positions:
                        size = validation['adjusted_size']
                        risk_manager.open_position(pair, signal, size, current_price, confidence)
                        current_position = {
                            'entry_time': current_data.iloc[-1]['timestamp'],
                            'entry_price': current_price,
                            'side': signal
                        }
            
            # Update equity curve
            current_equity = risk_manager.get_equity({pair: current_price})
            equity_curve.append(current_equity)
            
            # Check drawdown
            if risk_manager.check_drawdown(current_equity):
                logger.warning(f"Maximum drawdown reached at step {i}")
                break
        
        # Close any remaining positions
        if pair in risk_manager.positions:
            final_price = data.iloc[-1]['close']
            close_result = risk_manager.close_position(pair, final_price)
            trades.append({
                'entry_time': data.iloc[-1]['timestamp'],
                'exit_time': data.iloc[-1]['timestamp'],
                'entry_price': risk_manager.positions.get(pair, {}).get('entry_price', final_price),
                'exit_price': final_price,
                'side': risk_manager.positions.get(pair, {}).get('side', 'long'),
                'pnl': close_result['pnl'],
                'pnl_pct': close_result['pnl_pct'],
                'exit_reason': 'end_of_backtest'
            })
        
        # Calculate performance metrics
        results = self._calculate_metrics(equity_curve, trades)
        results['trades'] = trades
        results['equity_curve'] = equity_curve
        
        return results
    
    def _calculate_metrics(self, equity_curve: List[float], 
                          trades: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Annualized return (assuming daily data)
        days = len(equity_array) / 1440  # Approximate days if 1min candles
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades)
            
            # Average win/loss
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) \
                          if avg_loss > 0 and losing_trades else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': equity_array[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': sum([t['pnl'] for t in trades])
        }


def load_models_and_backtest(client: GateIOClient, pair: str, interval: str = '15m'):
    """Load pre-trained models and run backtest."""
    import os
    from .data_pipeline import DataPipeline
    
    logger.info(f"Loading models and backtesting for {pair}")
    
    # Initialize components
    data_pipeline = DataPipeline(client)
    risk_manager = RiskManager(
        initial_capital=Config.BACKTEST_INITIAL_BALANCE,
        risk_per_trade=Config.RISK_PER_TRADE
    )
    
    # Fetch and prepare data
    logger.info("Fetching historical data...")
    df = data_pipeline.fetch_historical_data(pair, interval, limit=2000)
    
    if df.empty:
        logger.error(f"No data fetched for {pair}")
        return None
    
    df = data_pipeline.prepare_features(df)
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < Config.LOOKBACK_WINDOW + 100:
        logger.error(f"Insufficient data for {pair}")
        return None
    
    # Get feature columns
    feature_cols = data_pipeline.get_feature_columns()
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_size = len(feature_cols)
    
    logger.info(f"Using {feature_size} features for {len(df)} candles")
    
    # Load pre-trained models
    lstm_path = os.path.join(Config.MODEL_PATH, f"{pair}_lstm.pt")
    transformer_path = os.path.join(Config.MODEL_PATH, f"{pair}_transformer.pt")
    
    lstm_model = None
    transformer_model = None
    
    # Load LSTM model
    if os.path.exists(lstm_path):
        logger.info(f"Loading LSTM model from {lstm_path}")
        lstm_model = LSTMModel(
            input_size=feature_size,
            hidden_size=Config.LSTM_UNITS
        )
        lstm_model.load(lstm_path)
    else:
        logger.warning(f"LSTM model not found at {lstm_path}, skipping LSTM predictions")
    
    # Load Transformer model
    if os.path.exists(transformer_path):
        logger.info(f"Loading Transformer model from {transformer_path}")
        transformer_model = TransformerModel(
            input_size=feature_size,
            d_model=Config.TRANSFORMER_D_MODEL,
            nhead=Config.TRANSFORMER_NHEAD,
            num_layers=Config.TRANSFORMER_NLAYERS
        )
        transformer_model.load(transformer_path)
    else:
        logger.warning(f"Transformer model not found at {transformer_path}, skipping Transformer predictions")
    
    if lstm_model is None and transformer_model is None:
        logger.error(f"No trained models found for {pair}. Please train models first using 'make train'")
        return None
    
    # Create ensemble strategy
    strategy = EnsembleStrategy(
        feature_size=feature_size,
        sequence_length=Config.LOOKBACK_WINDOW,
        lstm_model=lstm_model,
        transformer_model=transformer_model,
        rl_agent=None
    )
    
    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(
        initial_capital=Config.BACKTEST_INITIAL_BALANCE,
        commission=Config.BACKTEST_COMMISSION
    )
    
    results = backtester.run_backtest(df, strategy, risk_manager, pair)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results['annualized_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info("="*50 + "\n")
    
    return results


def train_models_and_backtest(client: GateIOClient, pair: str, interval: str = '15m'):
    """Train models and run backtest (legacy function, kept for compatibility)."""
    from .data_pipeline import DataPipeline
    
    logger.info(f"Training models and backtesting for {pair}")
    
    # Initialize components
    data_pipeline = DataPipeline(client)
    risk_manager = RiskManager(
        initial_capital=Config.BACKTEST_INITIAL_BALANCE,
        risk_per_trade=Config.RISK_PER_TRADE
    )
    
    # Fetch and prepare data
    logger.info("Fetching historical data...")
    df = data_pipeline.fetch_historical_data(pair, interval, limit=2000)
    
    if df.empty:
        logger.error(f"No data fetched for {pair}")
        return None
    
    df = data_pipeline.prepare_features(df)
    
    # Calculate target
    df['target'] = data_pipeline.calculate_target(df, Config.PREDICTION_HORIZON)
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < Config.LOOKBACK_WINDOW + 100:
        logger.error(f"Insufficient data for {pair}")
        return None
    
    # Split data (80% train, 20% test/backtest)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Training on {len(train_df)} samples, backtesting on {len(test_df)} samples")
    
    # Prepare sequences
    feature_cols = data_pipeline.get_feature_columns()
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    X_train, y_train = data_pipeline.create_sequences(
        train_df, feature_cols, 'target', Config.LOOKBACK_WINDOW, Config.PREDICTION_HORIZON
    )
    
    X_test, y_test = data_pipeline.create_sequences(
        test_df, feature_cols, 'target', Config.LOOKBACK_WINDOW, Config.PREDICTION_HORIZON
    )
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Could not create sequences")
        return None
    
    logger.info(f"Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=Config.LSTM_UNITS
    )
    lstm_model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Train Transformer model
    logger.info("Training Transformer model...")
    transformer_model = TransformerModel(
        input_size=len(feature_cols),
        d_model=Config.TRANSFORMER_D_MODEL,
        nhead=Config.TRANSFORMER_NHEAD,
        num_layers=Config.TRANSFORMER_NLAYERS
    )
    transformer_model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Train RL agent (simplified - would need proper environment setup)
    logger.info("Training RL agent...")
    # Note: RL training would require proper environment setup
    # For now, we'll skip it or use a simpler approach
    rl_agent = None
    
    # Create ensemble strategy
    strategy = EnsembleStrategy(
        feature_size=len(feature_cols),
        sequence_length=Config.LOOKBACK_WINDOW,
        lstm_model=lstm_model,
        transformer_model=transformer_model,
        rl_agent=rl_agent
    )
    
    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(
        initial_capital=Config.BACKTEST_INITIAL_BALANCE,
        commission=Config.BACKTEST_COMMISSION
    )
    
    results = backtester.run_backtest(test_df, strategy, risk_manager, pair)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results['annualized_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info("="*50 + "\n")
    
    return results


if __name__ == '__main__':
    from .config import Config
    from .gateio_client import GateIOClient
    
    Config.ensure_directories()
    
    # Initialize client (for backtesting, can use empty credentials)
    client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)
    
    # Run backtest for each pair (using pre-trained models)
    for pair in Config.TRADING_PAIRS:
        try:
            results = load_models_and_backtest(client, pair)
            if results:
                logger.info(f"Backtest completed for {pair}")
        except Exception as e:
            logger.error(f"Error backtesting {pair}: {e}")
            import traceback
            traceback.print_exc()

