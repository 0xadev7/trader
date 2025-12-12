"""Ensemble strategy combining multiple AI approaches."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from config import Config
from models import LSTMModel, TransformerModel
from rl_agent import PPOAgent, TradingEnv


class EnsembleStrategy:
    """Combines predictions from multiple AI models for robust trading decisions."""
    
    def __init__(self, feature_size: int, sequence_length: int,
                 lstm_model: Optional[LSTMModel] = None,
                 transformer_model: Optional[TransformerModel] = None,
                 rl_agent: Optional[PPOAgent] = None):
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.rl_agent = rl_agent
        
        # Ensemble weights (can be optimized)
        self.weights = {
            'lstm': 0.35,
            'transformer': 0.35,
            'rl': 0.30
        }
        
        # Performance tracking for dynamic weighting
        self.model_performance = {
            'lstm': {'wins': 0, 'losses': 0, 'returns': []},
            'transformer': {'wins': 0, 'losses': 0, 'returns': []},
            'rl': {'wins': 0, 'losses': 0, 'returns': []}
        }
    
    def predict(self, X: np.ndarray, prices: np.ndarray) -> Dict[str, any]:
        """Generate ensemble prediction.
        
        Args:
            X: Feature sequences (batch_size, sequence_length, features)
            prices: Current prices (batch_size,)
        
        Returns:
            Dict with 'signal' (str: 'buy', 'sell', 'hold'),
                    'confidence' (float: 0-1),
                    'predicted_return' (float),
                    'position_size' (float)
        """
        predictions = {}
        confidences = {}
        
        # LSTM prediction
        if self.lstm_model is not None:
            try:
                lstm_pred = self.lstm_model.predict(X)
                predictions['lstm'] = np.mean(lstm_pred)
                
                # Calculate confidence based on prediction magnitude
                pred_std = np.std(lstm_pred)
                confidences['lstm'] = min(abs(predictions['lstm']) / (pred_std + 1e-8), 1.0)
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")
                predictions['lstm'] = 0.0
                confidences['lstm'] = 0.0
        
        # Transformer prediction
        if self.transformer_model is not None:
            try:
                transformer_pred = self.transformer_model.predict(X)
                predictions['transformer'] = np.mean(transformer_pred)
                
                pred_std = np.std(transformer_pred)
                confidences['transformer'] = min(abs(predictions['transformer']) / (pred_std + 1e-8), 1.0)
            except Exception as e:
                logger.error(f"Transformer prediction error: {e}")
                predictions['transformer'] = 0.0
                confidences['transformer'] = 0.0
        
        # RL agent prediction
        if self.rl_agent is not None:
            try:
                # Use last state from sequence
                last_state = X[-1] if len(X.shape) > 2 else X
                
                # Add account features (placeholder - would come from actual account state)
                account_features = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
                full_state = np.concatenate([last_state, account_features])
                
                action, value, _ = self.rl_agent.select_action(full_state)
                
                # Convert action to signal
                action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                predictions['rl'] = {'action': action_map[action], 'value': value}
                confidences['rl'] = min(abs(value) / 10.0, 1.0)
            except Exception as e:
                logger.error(f"RL prediction error: {e}")
                predictions['rl'] = {'action': 'hold', 'value': 0.0}
                confidences['rl'] = 0.0
        
        # Ensemble prediction
        ensemble_signal, ensemble_confidence, predicted_return = self._combine_predictions(
            predictions, confidences
        )
        
        return {
            'signal': ensemble_signal,
            'confidence': ensemble_confidence,
            'predicted_return': predicted_return,
            'individual_predictions': predictions
        }
    
    def _combine_predictions(self, predictions: Dict, confidences: Dict) -> Tuple[str, float, float]:
        """Combine individual model predictions into ensemble decision."""
        signals = []
        weighted_returns = []
        total_weight = 0.0
        
        # Process LSTM and Transformer predictions (regression)
        for model_name in ['lstm', 'transformer']:
            if model_name in predictions:
                pred_value = predictions[model_name]
                confidence = confidences.get(model_name, 0.5)
                weight = self.weights.get(model_name, 0.0) * confidence
                
                if pred_value > 0.01:  # Threshold for buy
                    signals.append(('buy', weight))
                elif pred_value < -0.01:  # Threshold for sell
                    signals.append(('sell', weight))
                else:
                    signals.append(('hold', weight))
                
                weighted_returns.append(pred_value * weight)
                total_weight += weight
        
        # Process RL prediction (discrete action)
        if 'rl' in predictions:
            rl_pred = predictions['rl']
            rl_action = rl_pred['action']
            confidence = confidences.get('rl', 0.5)
            weight = self.weights.get('rl', 0.0) * confidence
            
            signals.append((rl_action, weight))
            weighted_returns.append(rl_pred['value'] * weight * 0.01)  # Scale value
            total_weight += weight
        
        # Calculate weighted signal
        signal_weights = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        for signal, weight in signals:
            signal_weights[signal] += weight
        
        # Determine final signal
        if signal_weights['buy'] > signal_weights['sell'] and signal_weights['buy'] > 0.4:
            final_signal = 'buy'
        elif signal_weights['sell'] > signal_weights['buy'] and signal_weights['sell'] > 0.4:
            final_signal = 'sell'
        else:
            final_signal = 'hold'
        
        # Calculate ensemble confidence
        final_confidence = min(total_weight, 1.0)
        
        # Calculate weighted predicted return
        predicted_return = sum(weighted_returns) / (total_weight + 1e-8)
        
        return final_signal, final_confidence, predicted_return
    
    def update_model_performance(self, model_name: str, pnl: float):
        """Update performance tracking for a model."""
        if model_name in self.model_performance:
            if pnl > 0:
                self.model_performance[model_name]['wins'] += 1
            else:
                self.model_performance[model_name]['losses'] += 1
            
            self.model_performance[model_name]['returns'].append(pnl)
            
            # Keep only recent returns
            if len(self.model_performance[model_name]['returns']) > 100:
                self.model_performance[model_name]['returns'] = \
                    self.model_performance[model_name]['returns'][-100:]
    
    def update_weights(self):
        """Dynamically update ensemble weights based on recent performance."""
        for model_name in ['lstm', 'transformer', 'rl']:
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                returns = perf['returns']
                
                if len(returns) > 10:
                    # Calculate Sharpe-like metric
                    mean_return = np.mean(returns)
                    std_return = np.std(returns) + 1e-8
                    sharpe = mean_return / std_return
                    
                    # Update weight based on performance
                    base_weight = 1.0 / 3.0
                    performance_multiplier = 1.0 + sharpe * 0.5
                    new_weight = base_weight * performance_multiplier
                    
                    # Normalize
                    self.weights[model_name] = max(0.1, min(0.6, new_weight))
        
        # Renormalize weights
        total_weight = sum(self.weights.values())
        for model_name in self.weights:
            self.weights[model_name] /= total_weight
        
        logger.info(f"Updated ensemble weights: {self.weights}")

