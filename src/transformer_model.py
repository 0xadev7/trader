"""Transformer model for price prediction."""

import torch
import torch.nn as nn
import numpy as np
import os
from loguru import logger
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer-based price prediction model."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 100,
    ):
        super(TransformerPredictor, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, sequence_length, features)

        Returns:
            Prediction tensor (batch_size, 1)
        """
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding (transpose for positional encoding)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Apply layer norm
        x = self.layer_norm(x)

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Use last timestep
        last_output = encoded[:, -1, :]

        # Project to output
        output = self.output_projection(last_output)

        return output


class TransformerModel:
    """Wrapper class for Transformer model with training and prediction."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = TransformerPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        self.criterion = nn.MSELoss()
        self.scaler = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> dict:
        """Train the model."""
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        # Normalize targets
        if self.scaler is None:
            self.scaler = StandardScaler()
            y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)).flatten()
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Learning rate scheduler
        # Note: verbose parameter not available in all PyTorch versions
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        predictions = self.model(X_batch).squeeze()
                        loss = self.criterion(predictions, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                # Update learning rate
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}"
                    )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

            # Inverse transform if scaler is available
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()

            return predictions

    def save(self, path: str):
        """Save model and scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler": self.scaler,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and scaler."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler = checkpoint.get("scaler")
        logger.info(f"Model loaded from {path}")
