"""Script to train all models for the trading bot."""

import os
from loguru import logger

from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .config import Config
from .gateio_client import GateIOClient
from .data_pipeline import DataPipeline


def train_models_for_pair(client: GateIOClient, pair: str, interval: str = "15m"):
    """Train LSTM and Transformer models for a trading pair."""
    logger.info(f"Training models for {pair}")

    # Initialize data pipeline
    data_pipeline = DataPipeline(client)

    # Fetch historical data
    logger.info(f"Fetching historical data for {pair}...")
    df = data_pipeline.fetch_historical_data(pair, interval, limit=3000)

    if df.empty:
        logger.error(f"No data fetched for {pair}")
        return False

    # Prepare features
    logger.info("Preparing features...")
    df = data_pipeline.prepare_features(df)

    # Calculate target (future returns)
    df["target"] = data_pipeline.calculate_target(df, Config.PREDICTION_HORIZON)
    df = df.dropna().reset_index(drop=True)

    if len(df) < Config.LOOKBACK_WINDOW + 200:
        logger.error(f"Insufficient data for {pair}")
        return False

    logger.info(f"Prepared {len(df)} samples with features")

    # Get feature columns
    feature_cols = data_pipeline.get_feature_columns()
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_size = len(feature_cols)

    logger.info(f"Using {feature_size} features")

    # Split data (80% train, 20% validation)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(
        f"Training on {len(train_df)} samples, validating on {len(val_df)} samples"
    )

    # Create sequences
    logger.info("Creating sequences...")
    X_train, y_train = data_pipeline.create_sequences(
        train_df,
        feature_cols,
        "target",
        Config.LOOKBACK_WINDOW,
        Config.PREDICTION_HORIZON,
    )

    X_val, y_val = data_pipeline.create_sequences(
        val_df,
        feature_cols,
        "target",
        Config.LOOKBACK_WINDOW,
        Config.PREDICTION_HORIZON,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        logger.error("Could not create sequences")
        return False

    logger.info(
        f"Created {len(X_train)} training and {len(X_val)} validation sequences"
    )

    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model = LSTMModel(
        input_size=feature_size,
        hidden_size=Config.LSTM_UNITS,
        num_layers=2,
        dropout=Config.DROPOUT_RATE,
        learning_rate=0.001,
    )

    lstm_history = lstm_model.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=15
    )

    # Save LSTM model
    lstm_path = os.path.join(Config.MODEL_PATH, f"{pair}_lstm.pt")
    lstm_model.save(lstm_path)
    logger.info(f"LSTM model saved to {lstm_path}")

    # Train Transformer model
    logger.info("Training Transformer model...")
    transformer_model = TransformerModel(
        input_size=feature_size,
        d_model=Config.TRANSFORMER_D_MODEL,
        nhead=Config.TRANSFORMER_NHEAD,
        num_layers=Config.TRANSFORMER_NLAYERS,
        dropout=Config.DROPOUT_RATE,
        learning_rate=0.001,
    )

    transformer_history = transformer_model.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=15
    )

    # Save Transformer model
    transformer_path = os.path.join(Config.MODEL_PATH, f"{pair}_transformer.pt")
    transformer_model.save(transformer_path)
    logger.info(f"Transformer model saved to {transformer_path}")

    logger.info(f"✅ Successfully trained models for {pair}")
    return True


if __name__ == "__main__":
    Config.ensure_directories()

    # Initialize client
    client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET)

    # Train models for each trading pair
    for pair in Config.TRADING_PAIRS:
        try:
            success = train_models_for_pair(client, pair)
            if success:
                logger.info(f"✅ Completed training for {pair}\n")
            else:
                logger.warning(f"⚠️ Failed to train models for {pair}\n")
        except Exception as e:
            logger.error(f"❌ Error training models for {pair}: {e}")
            import traceback

            traceback.print_exc()
