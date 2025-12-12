"""Main trading bot orchestrator."""

import os

import time
import schedule
from datetime import datetime
from loguru import logger

from .config import Config
from .gateio_client import GateIOClient, TradingPair
from .data_pipeline import DataPipeline
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .ensemble_strategy import EnsembleStrategy
from .risk_manager import RiskManager


# Configure logging
logger.add(
    os.path.join(Config.LOG_PATH, "trading_bot_{time}.log"),
    rotation="1 day",
    retention="30 days",
    level="INFO",
)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self):
        Config.ensure_directories()

        # Initialize Gate.io client
        if not Config.GATE_API_KEY or not Config.GATE_API_SECRET:
            raise ValueError(
                "Gate.io API credentials not configured. Please set GATE_API_KEY and GATE_API_SECRET in .env"
            )

        self.client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)

        # Initialize components
        self.data_pipeline = DataPipeline(self.client)
        self.risk_manager = RiskManager(
            initial_capital=Config.INITIAL_CAPITAL,
            risk_per_trade=Config.RISK_PER_TRADE,
            max_position_size=Config.MAX_POSITION_SIZE,
        )

        # Load or train models
        self.models = {}
        self.strategy = None

        # Trading state
        self.running = False
        self.last_update = {}

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
            self.strategy = EnsembleStrategy(
                feature_size=feature_size,
                sequence_length=Config.LOOKBACK_WINDOW,
                lstm_model=lstm_model,
                transformer_model=transformer_model,
                rl_agent=None,  # RL agent would need separate loading
            )

            self.models[pair] = {
                "lstm": lstm_model,
                "transformer": transformer_model,
                "strategy": self.strategy,
            }

            return True
        except Exception as e:
            logger.error(f"Error loading models for {pair}: {e}")
            return False

    def get_latest_data(self, pair: str) -> tuple:
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

    def generate_signal(self, pair: str, df: "pd.DataFrame") -> dict:
        """Generate trading signal for a pair."""
        if pair not in self.models or self.models[pair]["strategy"] is None:
            logger.warning(f"No strategy loaded for {pair}")
            return {"signal": "hold", "confidence": 0.0}

        try:
            strategy = self.models[pair]["strategy"]
            feature_cols = self.data_pipeline.get_feature_columns()
            feature_cols = [col for col in feature_cols if col in df.columns]

            # Prepare sequence
            feature_data = df[feature_cols].values
            if len(feature_data) < Config.LOOKBACK_WINDOW:
                return {"signal": "hold", "confidence": 0.0}

            X_seq = feature_data[-Config.LOOKBACK_WINDOW :].reshape(
                1, Config.LOOKBACK_WINDOW, -1
            )
            prices = df["close"].values[-Config.LOOKBACK_WINDOW :]

            # Get prediction
            prediction = strategy.predict(X_seq, prices[-1:])

            return prediction
        except Exception as e:
            logger.error(f"Error generating signal for {pair}: {e}")
            return {"signal": "hold", "confidence": 0.0}

    def execute_trade(
        self, pair: str, signal: str, current_price: float, confidence: float
    ):
        """Execute a trade based on signal."""
        try:
            # Check existing position
            if pair in self.risk_manager.positions:
                position = self.risk_manager.positions[pair]
                current_side = position["side"]

                # Check if we should close
                sl_tp = self.risk_manager.check_stop_loss_take_profit(
                    pair, current_price
                )
                if sl_tp:
                    logger.info(f"Closing {pair} position due to {sl_tp}")
                    self.close_position(pair, current_price, reason=sl_tp)
                    return

                # Check if we should reverse
                if signal != current_side and signal != "hold":
                    logger.info(
                        f"Reversing {pair} position from {current_side} to {signal}"
                    )
                    self.close_position(pair, current_price, reason="reverse")

            # Open new position if signal is strong enough
            if signal in ["buy", "sell"] and confidence > 0.6:
                if pair not in self.risk_manager.positions:
                    # Validate trade
                    position_size = self.risk_manager.calculate_position_size(
                        pair, current_price, confidence
                    )

                    validation = self.risk_manager.validate_trade(
                        pair, signal, position_size, current_price
                    )

                    if validation["allowed"]:
                        # Place order
                        converted_pair = self.client.convert_pair_format(pair)

                        try:
                            # Use market order for faster execution
                            order = self.client.place_order(
                                pair=converted_pair,
                                side=signal,
                                amount=validation["adjusted_size"],
                                order_type="market",
                            )

                            logger.info(f"Placed {signal} order for {pair}: {order}")

                            # Record position
                            self.risk_manager.open_position(
                                pair,
                                signal,
                                validation["adjusted_size"],
                                current_price,
                                confidence,
                            )
                        except Exception as e:
                            logger.error(f"Error placing order for {pair}: {e}")
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")

    def close_position(self, pair: str, current_price: float, reason: str = "signal"):
        """Close an open position."""
        try:
            if pair not in self.risk_manager.positions:
                return

            position = self.risk_manager.positions[pair]
            side = "sell" if position["side"] == "buy" else "buy"

            # Place closing order
            converted_pair = self.client.convert_pair_format(pair)

            try:
                order = self.client.place_order(
                    pair=converted_pair,
                    side=side,
                    amount=position["size"],
                    order_type="market",
                )

                logger.info(f"Placed closing {side} order for {pair}: {order}")

                # Record closure
                result = self.risk_manager.close_position(pair, current_price)
                logger.info(
                    f"Closed {pair} position: P&L = ${result['pnl']:.2f} ({result['pnl_pct']:.2%})"
                )
            except Exception as e:
                logger.error(f"Error closing position for {pair}: {e}")
        except Exception as e:
            logger.error(f"Error in close_position for {pair}: {e}")

    def update_balances(self):
        """Update account balances from exchange."""
        try:
            balances = self.client.get_account_balance()
            total_equity = 0.0

            for balance in balances:
                if balance["currency"] == Config.BASE_CURRENCY:
                    total_equity += float(balance.get("available", 0)) + float(
                        balance.get("locked", 0)
                    )

            # Update risk manager capital
            self.risk_manager.current_capital = total_equity
            logger.info(f"Updated balance: ${total_equity:,.2f} {Config.BASE_CURRENCY}")
        except Exception as e:
            logger.error(f"Error updating balances: {e}")

    def process_pair(self, pair: str):
        """Process a single trading pair."""
        try:
            logger.info(f"Processing {pair}...")

            # Get latest data
            df, latest_candle = self.get_latest_data(pair)
            if df is None or latest_candle is None:
                return

            current_price = latest_candle["close"]

            # Generate signal
            prediction = self.generate_signal(pair, df)
            signal = prediction.get("signal", "hold")
            confidence = prediction.get("confidence", 0.0)

            logger.info(
                f"{pair}: Signal={signal}, Confidence={confidence:.2f}, Price=${current_price:.2f}"
            )

            # Execute trade
            self.execute_trade(pair, signal, current_price, confidence)

            # Update last update time
            self.last_update[pair] = datetime.now()
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            import traceback

            traceback.print_exc()

    def run_iteration(self):
        """Run one iteration of the trading loop."""
        logger.info("Starting trading iteration...")

        # Update balances
        self.update_balances()

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
                self.process_pair(pair)

                # Small delay between pairs
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in run_iteration for {pair}: {e}")

        logger.info("Trading iteration completed")

    def start(self):
        """Start the trading bot."""
        logger.info("Starting AI Trading Bot...")
        logger.info(f"Trading pairs: {Config.TRADING_PAIRS}")
        logger.info(f"Initial capital: ${Config.INITIAL_CAPITAL:,.2f}")

        # Load models for all pairs
        for pair in Config.TRADING_PAIRS:
            logger.info(f"Loading models for {pair}...")
            self.load_models(pair)

        self.running = True

        # Schedule trading iterations
        schedule.every(5).minutes.do(self.run_iteration)

        logger.info("Trading bot started. Running every 5 minutes...")
        logger.warning("LIVE TRADING ENABLED - Monitor carefully!")

        # Main loop
        while self.running:
            schedule.run_pending()
            time.sleep(10)  # Check every 10 seconds

    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self.running = False


if __name__ == "__main__":
    import signal
    import sys

    bot = TradingBot()

    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        bot.stop()
