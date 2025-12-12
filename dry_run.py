"""Dry-run mode for simulating trading without real orders."""
import time
import schedule
from datetime import datetime, timedelta
from loguru import logger
from config import Config
from gateio_client import GateIOClient
from data_pipeline import DataPipeline
from models import LSTMModel, TransformerModel
from ensemble_strategy import EnsembleStrategy
from risk_manager import RiskManager
import os
import pandas as pd
import json


class DryRunSimulator:
    """Simulates trading without placing real orders."""
    
    def __init__(self, duration_days: int = 7):
        Config.ensure_directories()
        
        self.duration_days = duration_days
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(days=duration_days)
        
        # Initialize client (for data fetching only, not trading)
        # Note: Dry-run needs real API keys to fetch market data
        api_key = Config.GATE_API_KEY or ''
        api_secret = Config.GATE_API_SECRET or ''
        if not api_key or not api_secret:
            logger.warning("API credentials not set. Dry-run needs API keys to fetch market data.")
        self.client = GateIOClient(api_key, api_secret)
        
        # Initialize components
        self.data_pipeline = DataPipeline(self.client)
        self.risk_manager = RiskManager(
            initial_capital=Config.INITIAL_CAPITAL,
            risk_per_trade=Config.RISK_PER_TRADE,
            max_position_size=Config.MAX_POSITION_SIZE,
            stop_loss_pct=Config.STOP_LOSS_PCT,
            take_profit_pct=Config.TAKE_PROFIT_PCT,
            max_drawdown_pct=Config.MAX_DRAWDOWN_PCT
        )
        
        # Models
        self.models = {}
        
        # Trade log
        self.trade_log = []
        self.equity_history = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_equity': Config.INITIAL_CAPITAL,
            'min_equity': Config.INITIAL_CAPITAL,
            'start_equity': Config.INITIAL_CAPITAL
        }
        
        self.running = False
    
    def load_models(self, pair: str) -> bool:
        """Load pre-trained models for a trading pair."""
        try:
            feature_cols = self.data_pipeline.get_feature_columns()
            feature_size = len(feature_cols)
            
            lstm_path = os.path.join(Config.MODEL_PATH, f"{pair}_lstm.pt")
            transformer_path = os.path.join(Config.MODEL_PATH, f"{pair}_transformer.pt")
            
            lstm_model = None
            transformer_model = None
            
            # Load LSTM
            if os.path.exists(lstm_path):
                lstm_model = LSTMModel(input_size=feature_size)
                lstm_model.load(lstm_path)
                logger.info(f"Loaded LSTM model for {pair}")
            
            # Load Transformer
            if os.path.exists(transformer_path):
                transformer_model = TransformerModel(input_size=feature_size)
                transformer_model.load(transformer_path)
                logger.info(f"Loaded Transformer model for {pair}")
            
            # Create ensemble strategy
            strategy = EnsembleStrategy(
                feature_size=feature_size,
                sequence_length=Config.LOOKBACK_WINDOW,
                lstm_model=lstm_model,
                transformer_model=transformer_model,
                rl_agent=None
            )
            
            self.models[pair] = {
                'lstm': lstm_model,
                'transformer': transformer_model,
                'strategy': strategy
            }
            
            return True
        except Exception as e:
            logger.error(f"Error loading models for {pair}: {e}")
            return False
    
    def get_latest_data(self, pair: str):
        """Get latest market data for a trading pair."""
        try:
            df = self.data_pipeline.fetch_historical_data(
                pair, Config.PRIMARY_TIMEFRAME, limit=Config.LOOKBACK_WINDOW + 50
            )
            
            if df.empty:
                logger.warning(f"No data fetched for {pair}")
                return None, None
            
            df = self.data_pipeline.prepare_features(df)
            df = df.dropna().reset_index(drop=True)
            
            if len(df) < Config.LOOKBACK_WINDOW:
                logger.warning(f"Insufficient data for {pair}")
                return None, None
            
            return df, df.iloc[-1]
        except Exception as e:
            logger.error(f"Error getting data for {pair}: {e}")
            return None, None
    
    def generate_signal(self, pair: str, df: pd.DataFrame) -> dict:
        """Generate trading signal for a pair."""
        if pair not in self.models or self.models[pair]['strategy'] is None:
            logger.warning(f"No strategy loaded for {pair}")
            return {'signal': 'hold', 'confidence': 0.0}
        
        try:
            strategy = self.models[pair]['strategy']
            feature_cols = self.data_pipeline.get_feature_columns()
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            # Prepare sequence
            feature_data = df[feature_cols].values
            if len(feature_data) < Config.LOOKBACK_WINDOW:
                return {'signal': 'hold', 'confidence': 0.0}
            
            X_seq = feature_data[-Config.LOOKBACK_WINDOW:].reshape(1, Config.LOOKBACK_WINDOW, -1)
            prices = df['close'].values[-Config.LOOKBACK_WINDOW:]
            
            # Get prediction
            prediction = strategy.predict(X_seq, prices[-1:])
            
            return prediction
        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}")
            return {'signal': 'hold', 'confidence': 0.0}
    
    def simulate_trade(self, pair: str, signal: str, current_price: float, confidence: float, timestamp: datetime):
        """Simulate a trade without placing real order."""
        try:
            # Check existing position
            if pair in self.risk_manager.positions:
                position = self.risk_manager.positions[pair]
                
                # Check stop-loss / take-profit
                sl_tp = self.risk_manager.check_stop_loss_take_profit(pair, current_price)
                if sl_tp:
                    self.simulate_close_position(pair, current_price, timestamp, reason=sl_tp)
                    return
                
                # Check if we should reverse
                if signal != position['side'] and signal != 'hold':
                    self.simulate_close_position(pair, current_price, timestamp, reason='reverse')
            
            # Open new position if signal is strong enough
            if signal in ['buy', 'sell'] and confidence > 0.6:
                if pair not in self.risk_manager.positions:
                    # Validate trade
                    position_size = self.risk_manager.calculate_position_size(pair, current_price, confidence)
                    
                    validation = self.risk_manager.validate_trade(
                        pair, signal, position_size, current_price
                    )
                    
                    if validation['allowed']:
                        # Simulate opening position (no real order)
                        self.risk_manager.open_position(
                            pair, signal, validation['adjusted_size'],
                            current_price, confidence
                        )
                        
                        logger.info(f"[DRY-RUN] {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                                  f"OPEN {signal.upper()} {pair}: "
                                  f"Size={validation['adjusted_size']:.6f}, "
                                  f"Price=${current_price:.2f}, "
                                  f"Confidence={confidence:.2f}")
                        
                        self.trade_log.append({
                            'timestamp': timestamp.isoformat(),
                            'pair': pair,
                            'action': 'open',
                            'side': signal,
                            'price': current_price,
                            'size': validation['adjusted_size'],
                            'confidence': confidence
                        })
        except Exception as e:
            logger.error(f"Error simulating trade for {pair}: {e}")
    
    def simulate_close_position(self, pair: str, current_price: float, timestamp: datetime, reason: str = 'signal'):
        """Simulate closing a position."""
        try:
            if pair not in self.risk_manager.positions:
                return
            
            position = self.risk_manager.positions[pair]
            
            # Record closure
            result = self.risk_manager.close_position(pair, current_price)
            
            # Update statistics
            self.stats['total_trades'] += 1
            if result['pnl'] > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            self.stats['total_pnl'] += result['pnl']
            
            logger.info(f"[DRY-RUN] {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                      f"CLOSE {position['side'].upper()} {pair}: "
                      f"Entry=${position['entry_price']:.2f}, "
                      f"Exit=${current_price:.2f}, "
                      f"P&L=${result['pnl']:.2f} ({result['pnl_pct']:.2%}), "
                      f"Reason={reason}")
            
            self.trade_log.append({
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'action': 'close',
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': result['pnl'],
                'pnl_pct': result['pnl_pct'],
                'reason': reason
            })
        except Exception as e:
            logger.error(f"Error in simulate_close_position for {pair}: {e}")
    
    def process_pair(self, pair: str, timestamp: datetime):
        """Process a single trading pair."""
        try:
            # Get latest data
            df, latest_candle = self.get_latest_data(pair)
            if df is None or latest_candle is None:
                return
            
            current_price = latest_candle['close']
            
            # Generate signal
            prediction = self.generate_signal(pair, df)
            signal = prediction.get('signal', 'hold')
            confidence = prediction.get('confidence', 0.0)
            
            if signal != 'hold' and confidence > 0.5:
                logger.info(f"[DRY-RUN] {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                          f"{pair}: Signal={signal.upper()}, "
                          f"Confidence={confidence:.2f}, "
                          f"Price=${current_price:.2f}")
            
            # Simulate trade
            self.simulate_trade(pair, signal, current_price, confidence, timestamp)
            
            # Update equity tracking
            current_prices = {pair: current_price}
            equity = self.risk_manager.get_equity(current_prices)
            self.equity_history.append({
                'timestamp': timestamp.isoformat(),
                'equity': equity,
                'pair': pair,
                'price': current_price
            })
            
            # Update statistics
            if equity > self.stats['max_equity']:
                self.stats['max_equity'] = equity
            if equity < self.stats['min_equity']:
                self.stats['min_equity'] = equity
                
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
    
    def run_iteration(self, current_time: datetime):
        """Run one iteration of the simulation."""
        # Process each trading pair
        for pair in Config.TRADING_PAIRS:
            try:
                # Ensure models are loaded
                if pair not in self.models:
                    logger.info(f"Loading models for {pair}...")
                    if not self.load_models(pair):
                        logger.warning(f"Could not load models for {pair}, skipping")
                        continue
                
                # Process pair
                self.process_pair(pair, current_time)
                
                # Small delay between pairs
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in run_iteration for {pair}: {e}")
    
    def print_statistics(self):
        """Print final statistics."""
        final_equity = self.risk_manager.get_equity(
            {pair: 0 for pair in Config.TRADING_PAIRS}  # Close all at last known prices
        )
        
        # Close any remaining positions
        for pair in list(self.risk_manager.positions.keys()):
            # Get last known price from equity history
            last_price = 0
            for entry in reversed(self.equity_history):
                if entry['pair'] == pair:
                    last_price = entry['price']
                    break
            
            if last_price > 0:
                self.simulate_close_position(pair, last_price, datetime.now(), reason='end_of_dry_run')
        
        # Calculate final statistics
        duration = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        total_return = (final_equity - self.stats['start_equity']) / self.stats['start_equity']
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades'] 
                   if self.stats['total_trades'] > 0 else 0)
        max_drawdown = (self.stats['max_equity'] - self.stats['min_equity']) / self.stats['max_equity']
        
        logger.info("\n" + "="*60)
        logger.info("DRY-RUN SIMULATION RESULTS")
        logger.info("="*60)
        logger.info(f"Duration: {duration:.2f} hours ({self.duration_days} days)")
        logger.info(f"Start Equity: ${self.stats['start_equity']:,.2f}")
        logger.info(f"End Equity: ${final_equity:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Total P&L: ${self.stats['total_pnl']:,.2f}")
        logger.info(f"Total Trades: {self.stats['total_trades']}")
        logger.info(f"Winning Trades: {self.stats['winning_trades']}")
        logger.info(f"Losing Trades: {self.stats['losing_trades']}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Max Equity: ${self.stats['max_equity']:,.2f}")
        logger.info(f"Min Equity: ${self.stats['min_equity']:,.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info("="*60 + "\n")
        
        # Save results
        results = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_hours': duration,
            'start_equity': self.stats['start_equity'],
            'end_equity': final_equity,
            'total_return': total_return,
            'total_pnl': self.stats['total_pnl'],
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades': self.trade_log,
            'equity_history': self.equity_history
        }
        
        results_path = os.path.join(Config.LOG_PATH, f"dry_run_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def start(self):
        """Start dry-run simulation."""
        logger.info("="*60)
        logger.info("STARTING DRY-RUN SIMULATION")
        logger.info("="*60)
        logger.info(f"Duration: {self.duration_days} days")
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Initial Capital: ${Config.INITIAL_CAPITAL:,.2f}")
        logger.info(f"Trading Pairs: {', '.join(Config.TRADING_PAIRS)}")
        logger.info("="*60 + "\n")
        
        # Load models for all pairs
        for pair in Config.TRADING_PAIRS:
            logger.info(f"Loading models for {pair}...")
            self.load_models(pair)
        
        self.running = True
        
        # Calculate iteration interval (every 5 minutes)
        iteration_interval_minutes = 5
        total_iterations = (self.duration_days * 24 * 60) // iteration_interval_minutes
        
        logger.info(f"Running {total_iterations} iterations (every {iteration_interval_minutes} minutes)\n")
        
        current_time = self.start_time
        iteration = 0
        
        try:
            while current_time < self.end_time and self.running:
                iteration += 1
                
                logger.info(f"[Iteration {iteration}/{total_iterations}] "
                          f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run iteration
                self.run_iteration(current_time)
                
                # Advance time
                current_time += timedelta(minutes=iteration_interval_minutes)
                
                # Small delay for readability
                time.sleep(0.1)
                
                # Progress update
                if iteration % 10 == 0:
                    progress = (current_time - self.start_time).total_seconds() / \
                              (self.end_time - self.start_time).total_seconds() * 100
                    logger.info(f"Progress: {progress:.1f}%")
            
            logger.info("\nSimulation completed!")
            
        except KeyboardInterrupt:
            logger.info("\nSimulation interrupted by user")
        
        # Print final statistics
        self.print_statistics()


if __name__ == '__main__':
    import sys
    
    # Get duration from command line or use default
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    logger.info(f"Starting dry-run for {duration} days")
    
    simulator = DryRunSimulator(duration_days=duration)
    simulator.start()

