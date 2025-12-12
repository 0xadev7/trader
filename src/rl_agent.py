"""Reinforcement Learning agent using PPO for trading decisions."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from loguru import logger


class TradingEnv:
    """Trading environment for RL agent."""

    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000,
        commission: float = 0.001,
    ):
        self.data = data  # Feature data (n_samples, features)
        self.prices = prices  # Price data (n_samples,)
        self.initial_balance = initial_balance
        self.commission = commission

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.balance = self.initial_balance
        self.position = (
            0.0  # Position size (0 = no position, 1 = max long, -1 = max short)
        )
        self.equity = [self.initial_balance]
        self.current_step = 0
        self.max_steps = len(self.data) - 1

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        if self.current_step >= len(self.data):
            return np.zeros(self.data.shape[1] + 5)

        # Current features
        features = self.data[self.current_step].copy()

        # Account state
        current_price = self.prices[self.current_step]
        portfolio_value = self.balance + (self.position * self.balance * current_price)

        account_features = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Current position
                (portfolio_value - self.initial_balance)
                / self.initial_balance,  # Return
                len(self.equity) > 1
                and (self.equity[-1] - self.equity[-2]) / self.equity[-2]
                or 0,  # Equity change
                (
                    current_price / self.prices[max(0, self.current_step - 20)]
                    if self.current_step > 0
                    else 1.0
                ),  # Price momentum
            ]
        )

        return np.concatenate([features, account_features])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (state, reward, done, info).

        Actions:
            0: Hold
            1: Buy (increase long position)
            2: Sell (close long or open short)
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0, True, {}

        current_price = self.prices[self.current_step]
        prev_equity = self.equity[-1] if self.equity else self.initial_balance

        # Execute action
        if action == 1:  # Buy
            if self.position < 0.95:  # Max position
                trade_amount = 0.1  # 10% position size
                cost = (
                    trade_amount * self.balance * current_price * (1 + self.commission)
                )
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += trade_amount
        elif action == 2:  # Sell
            if self.position > -0.95:  # Max short position
                trade_amount = 0.1
                if self.position > 0:  # Close long
                    revenue = (
                        trade_amount
                        * self.balance
                        * current_price
                        * (1 - self.commission)
                    )
                    self.balance += revenue
                    self.position -= trade_amount
                else:  # Open short
                    revenue = (
                        trade_amount
                        * self.balance
                        * current_price
                        * (1 - self.commission)
                    )
                    self.balance += revenue
                    self.position -= trade_amount

        # Calculate new equity
        portfolio_value = self.balance + (self.position * self.balance * current_price)
        self.equity.append(portfolio_value)

        # Calculate reward (risk-adjusted return)
        equity_change = (
            (portfolio_value - prev_equity) / prev_equity if prev_equity > 0 else 0
        )

        # Sharpe ratio-like reward (encourages consistent returns)
        if len(self.equity) > 20:
            returns = np.diff(self.equity[-20:]) / np.array(self.equity[-20:-1])
            sharpe = (
                np.mean(returns) / (np.std(returns) + 1e-8)
                if np.std(returns) > 0
                else 0
            )
            reward = equity_change * 100 + sharpe * 0.1
        else:
            reward = equity_change * 100

        # Penalty for large drawdowns
        if len(self.equity) > 1:
            peak = max(self.equity)
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
            if drawdown > 0.15:  # 15% drawdown
                reward -= 10

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "equity": portfolio_value,
            "position": self.position,
            "return": (portfolio_value - self.initial_balance) / self.initial_balance,
        }

        return self._get_state(), reward, done, info


class PPOPolicy(nn.Module):
    """PPO Policy Network."""

    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256):
        super(PPOPolicy, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def act(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """Sample action from policy."""
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), value.item(), log_prob


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        device: str = "cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, value, log_prob = self.policy.act(state_tensor)
        return action, value, log_prob.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store transition in memory."""
        self.memory["states"].append(state)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["log_probs"].append(log_prob)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)

    def compute_returns(self) -> np.ndarray:
        """Compute discounted returns."""
        returns = []
        discounted_reward = 0

        for reward, done in zip(
            reversed(self.memory["rewards"]), reversed(self.memory["dones"])
        ):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        return np.array(returns)

    def update(self):
        """Update policy using PPO."""
        if len(self.memory["states"]) == 0:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory["states"])).to(self.device)
        actions = torch.LongTensor(self.memory["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory["log_probs"]).to(self.device)
        old_values = torch.FloatTensor(self.memory["values"]).to(self.device)

        # Compute returns and advantages
        returns = self.compute_returns()
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages = returns_tensor - old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy predictions
            action_probs, values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            # Compute value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Clear memory
        self.memory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

    def save(self, path: str):
        """Save agent."""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"PPO agent saved to {path}")

    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"PPO agent loaded from {path}")
