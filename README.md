# Advanced AI Trading Bot for Gate.io

A sophisticated algorithmic trading bot using state-of-the-art AI techniques including Reinforcement Learning (PPO), Deep Learning (LSTM/Transformer), and Ensemble methods. This bot combines multiple machine learning approaches to make robust trading decisions with comprehensive risk management.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
  - [Training Models](#training-models)
  - [Backtesting](#backtesting)
  - [Dry-Run Mode](#dry-run-mode)
  - [Live Trading](#live-trading)
- [Architecture Details](#architecture-details)
- [Risk Management](#risk-management)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Important Disclaimers](#important-disclaimers)

## Features

### AI-Powered Trading
- 🤖 **Reinforcement Learning**: PPO-based agent for adaptive trading decisions
- 🧠 **Deep Learning Models**: 
  - LSTM with bidirectional layers and attention mechanism
  - Transformer encoder for capturing long-range dependencies
- 📊 **Ensemble Strategy**: Combines multiple AI approaches with dynamic weighting
- 🎯 **Confidence-Based Decisions**: Position sizing adapts to prediction certainty

### Risk Management
- 🛡️ **Dynamic Position Sizing**: Based on risk per trade, confidence, and stop-loss distance
- 🚫 **Stop-Loss & Take-Profit**: Automatic position management
- 📉 **Maximum Drawdown Protection**: Halts trading if drawdown exceeds threshold
- ⚖️ **Per-Trade Risk Limits**: Configurable risk percentage per trade

### Testing & Validation
- 🔄 **Backtesting Framework**: Comprehensive historical data simulation
- 🧪 **Dry-Run Mode**: Simulate trading with real-time data without placing orders
- 📈 **Performance Metrics**: Sharpe ratio, win rate, drawdown, profit factor
- 📊 **Multi-Timeframe Analysis**: Analyze multiple timeframes simultaneously

### Developer Experience
- ⚙️ **Environment Configuration**: All parameters configurable via `.env` file
- 🛠️ **Makefile**: Simple commands for all common tasks
- 📝 **Extensive Logging**: All decisions logged for analysis
- 🔧 **Modular Architecture**: Easy to extend and customize

## Architecture Overview

The bot uses an ensemble approach combining three AI techniques:

1. **LSTM Model**: Bidirectional LSTM with self-attention for sequence prediction
2. **Transformer Model**: Transformer encoder for attention-based modeling
3. **Reinforcement Learning**: PPO agent learning optimal trading actions
4. **Ensemble Strategy**: Dynamically weights and combines all predictions

All models work together through a voting system that considers:
- Individual model predictions
- Confidence scores
- Historical performance
- Risk-adjusted position sizing

## Project Structure

```
trader/
├── config.py                 # Centralized configuration (reads from .env)
├── gateio_client.py          # Gate.io API client with authentication
├── data_pipeline.py          # Data collection and preprocessing
├── models/                   # AI models
│   ├── __init__.py
│   ├── lstm_model.py        # LSTM predictor with attention
│   └── transformer_model.py # Transformer predictor
├── rl_agent.py              # Reinforcement Learning (PPO) agent
├── ensemble_strategy.py     # Ensemble strategy combining models
├── risk_manager.py          # Risk management and position sizing
├── backtest.py              # Backtesting framework
├── dry_run.py               # Dry-run simulation mode
├── train_models.py          # Model training script
├── main.py                  # Main trading bot orchestrator
├── example_usage.py         # Usage examples
├── Makefile                 # Command interface
├── env.template             # Environment configuration template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

### Using Makefile (Recommended)

```bash
# 1. Setup environment
make setup          # Create .env file from template
make install        # Install dependencies

# 2. Configure
# Edit .env with your Gate.io API credentials

# 3. Train models
make train          # Train all models (10-30 min per pair)

# 4. Test strategy
make backtest       # Historical backtest
make dry-run        # Simulate with real-time data (7 days)

# 5. Run live trading (use with caution!)
make run            # Start live trading bot
```

### Manual Setup

See [Installation](#installation) section for detailed manual setup instructions.

## Installation

### Prerequisites

- Python 3.8 or higher
- Gate.io account with API credentials
- Internet connection for data fetching

### Step-by-Step Installation

1. **Clone or download the project**
   ```bash
   cd trader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or use Makefile:
   make install
   ```

3. **Create environment file**
   ```bash
   cp env.template .env
   # Or use Makefile:
   make setup
   ```

4. **Configure API credentials**
   
   Edit `.env` file and add your Gate.io API credentials:
   ```env
   GATE_API_KEY=your_api_key_here
   GATE_API_SECRET=your_api_secret_here
   ```
   
   **Important**: 
   - Get API keys from [Gate.io API v4](https://www.gate.io/en/apiv4)
   - For backtesting/dry-run: Read-only keys are sufficient
   - For live trading: Trading-enabled keys are required
   - Start with paper trading or small amounts

5. **Verify configuration**
   ```bash
   make check-env
   ```

## Configuration

All configuration is done through the `.env` file. A comprehensive template is provided in `env.template`.

### Essential Configuration

```env
# Gate.io API Credentials
GATE_API_KEY=your_api_key_here
GATE_API_SECRET=your_api_secret_here

# Trading Configuration
BASE_CURRENCY=USDT
TRADING_PAIRS=BTC_USDT,ETH_USDT
INITIAL_CAPITAL=1000

# Risk Management
RISK_PER_TRADE=0.02              # Risk 2% of capital per trade
MAX_POSITION_SIZE=0.1            # Maximum 10% of capital per position
STOP_LOSS_PCT=0.02               # 2% stop loss
TAKE_PROFIT_PCT=0.04             # 4% take profit
MAX_DRAWDOWN_PCT=0.15            # 15% maximum drawdown
```

### Trading Parameters

```env
PRIMARY_TIMEFRAME=15m            # Timeframe: 1m, 5m, 15m, 1h, 4h, 1d
LOOKBACK_WINDOW=100              # Number of candles for sequences
TRADING_INTERVAL_MINUTES=5       # How often to check markets
```

### Model Configuration

```env
LSTM_UNITS=128
TRANSFORMER_D_MODEL=128
TRANSFORMER_NHEAD=8
TRANSFORMER_NLAYERS=4
DROPOUT_RATE=0.2
```

### Backtesting Configuration

```env
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2024-01-01
BACKTEST_INITIAL_BALANCE=10000
BACKTEST_COMMISSION=0.001        # 0.1% commission
```

### Dry-Run Configuration

```env
DRY_RUN_DURATION_DAYS=7          # Duration in days
DRY_RUN_INTERVAL_MINUTES=5       # Check interval in minutes
```

### Complete Configuration Reference

See `env.template` for all available configuration options with detailed comments.

## Usage Guide

### Training Models

Before trading, you need to train the AI models on historical data.

**Using Makefile:**
```bash
make train
```

**Manual:**
```bash
python train_models.py
```

**What happens:**
- Fetches historical data (3000+ candles) for each trading pair
- Calculates technical indicators (50+ features)
- Splits data into training (80%) and validation (20%)
- Trains LSTM and Transformer models
- Saves models to `./models/` directory

**Duration:** 10-30 minutes per trading pair (depends on hardware)

**Requirements:**
- Sufficient historical data (2000+ candles recommended)
- GPU optional but recommended for faster training

### Backtesting

Test your strategy on historical data to validate performance.

**Using Makefile:**
```bash
make backtest
```

**Manual:**
```bash
python backtest.py
```

**What happens:**
- Loads or trains models
- Runs simulation on historical data
- Calculates performance metrics
- Displays results in console

**Metrics Provided:**
- Total return and annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Total number of trades

**When to use:**
- Initial strategy validation
- Parameter optimization
- Historical performance analysis
- Before live trading

### Dry-Run Mode

Simulate trading with **real-time market data** without placing actual orders. This is the recommended way to test your bot before live trading.

#### Overview

Dry-run mode allows you to:
- ✅ Test with current market conditions
- ✅ Validate bot behavior over days or weeks
- ✅ Understand decision-making patterns
- ✅ Build confidence before risking real capital
- ❌ Does NOT place real orders
- ❌ Does NOT use real capital

#### Quick Start

**Using Makefile (Recommended):**
```bash
make dry-run                    # Run for 7 days (default)
make dry-run dry-run-days=30    # Run for 30 days
```

**Using Python:**
```bash
python dry_run.py 7             # Run for 7 days
python dry_run.py 30            # Run for 30 days
```

#### What Gets Simulated

1. **Market Data Fetching**
   - Real-time price data from Gate.io
   - Historical candles for technical analysis
   - Current market conditions

2. **Signal Generation**
   - LSTM model predictions
   - Transformer model predictions
   - Ensemble strategy decisions
   - Confidence scores

3. **Trade Execution**
   - Position sizing based on risk parameters
   - Stop-loss and take-profit levels
   - Position opening and closing
   - P&L calculation

4. **Risk Management**
   - Maximum position size limits
   - Stop-loss triggers
   - Take-profit triggers
   - Maximum drawdown protection

#### Configuration

Add to your `.env` file:

```env
DRY_RUN_DURATION_DAYS=7
DRY_RUN_INTERVAL_MINUTES=5
```

#### Output

Dry-run provides comprehensive output:

**Console Output:**
- Real-time trade decisions
- Position openings and closings
- P&L for each trade
- Progress updates

**Statistics:**
- Total return
- Number of trades
- Win rate
- Maximum drawdown
- Average profit/loss per trade

**Results File:**
- Saved to: `logs/dry_run_results_YYYYMMDD_HHMMSS.json`
- Contains: Complete trade log, equity curve, statistics

#### Example Output

```
============================================================
STARTING DRY-RUN SIMULATION
============================================================
Duration: 7 days
Start Time: 2024-01-15 10:00:00
End Time: 2024-01-22 10:00:00
Initial Capital: $1,000.00
Trading Pairs: BTC_USDT, ETH_USDT
============================================================

[DRY-RUN] 2024-01-15 10:05:00 - OPEN BUY BTC_USDT: Size=0.001234, Price=$42000.00, Confidence=0.75
[DRY-RUN] 2024-01-15 10:30:00 - CLOSE BUY BTC_USDT: Entry=$42000.00, Exit=$42840.00, P&L=$1.04 (2.00%), Reason=take_profit

...

============================================================
DRY-RUN SIMULATION RESULTS
============================================================
Duration: 168.00 hours (7 days)
Start Equity: $1,000.00
End Equity: $1,045.50
Total Return: 4.55%
Total P&L: $45.50
Total Trades: 12
Winning Trades: 8
Losing Trades: 4
Win Rate: 66.67%
Max Equity: $1,052.30
Min Equity: $985.20
Max Drawdown: 6.38%
============================================================
```

#### Interpreting Results

**Good Signs ✅:**
- Positive total return
- Win rate > 50%
- Reasonable drawdown (< 15%)
- Consistent profit distribution
- Reasonable number of trades

**Warning Signs ⚠️:**
- Negative total return
- Win rate < 40%
- Very high drawdown (> 20%)
- Too few trades (possible overfitting)
- Too many trades (overtrading)

**What to Do:**

If results are good:
1. Extend dry-run to longer period (30+ days)
2. Test with different market conditions
3. Consider starting with small real capital

If results are poor:
1. Review and adjust risk parameters
2. Retrain models with more data
3. Adjust confidence thresholds
4. Try different trading pairs or timeframes
5. Review trading logic and indicators

#### Differences from Backtest

| Feature | Backtest | Dry-Run |
|---------|----------|---------|
| Data | Historical | Real-time |
| Speed | Fast (all at once) | Slower (real-time simulation) |
| Orders | Simulated | Simulated |
| Market Conditions | Past | Current |
| Best For | Strategy validation | Live testing |

#### Tips

1. **Start Short**: Test with 1-3 days first to ensure everything works
2. **Monitor Progress**: Watch console output to understand bot behavior
3. **Review Logs**: Check detailed logs for decision-making patterns
4. **Test Different Scenarios**: Run multiple dry-runs with different configurations
5. **Compare**: Run both backtest and dry-run to compare results

### Live Trading

**⚠️ WARNING: This will place REAL orders with REAL money!**

**Using Makefile:**
```bash
make run            # Includes confirmation prompt
```

**Manual:**
```bash
python main.py
```

**Before running live:**
1. ✅ Complete backtesting
2. ✅ Run dry-run simulation
3. ✅ Review results carefully
4. ✅ Verify all configuration settings
5. ✅ Start with small amounts
6. ✅ Monitor closely initially

**What happens:**
- Loads trained models
- Monitors market conditions (every 5 minutes by default)
- Generates trading signals
- Executes trades based on signals and risk rules
- Manages positions (stop-loss, take-profit)
- Logs all activities

**Safety Features:**
- Maximum drawdown protection (halts trading if exceeded)
- Position size limits
- Risk per trade limits
- Stop-loss on all positions
- Extensive logging

## Architecture Details

### Components

#### 1. Data Pipeline (`data_pipeline.py`)

**Responsibilities:**
- Fetch historical and real-time market data from Gate.io
- Calculate comprehensive technical indicators (50+ features)
- Create sequences for time-series models
- Normalize and preprocess data

**Key Features:**
- Multi-timeframe data support
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, OBV
- Price features: returns, volatility, momentum
- Sequence generation for LSTM/Transformer models

#### 2. AI Models

**LSTM Model (`models/lstm_model.py`):**
- Architecture: Bidirectional LSTM with attention mechanism
- Purpose: Predict future price movements based on historical patterns
- Features:
  - Multi-layer LSTM with dropout
  - Self-attention mechanism
  - Fully connected output layers
  - Robust to temporal dependencies

**Transformer Model (`models/transformer_model.py`):**
- Architecture: Transformer encoder with positional encoding
- Purpose: Capture long-range dependencies and complex patterns
- Features:
  - Multi-head self-attention
  - Positional encoding for sequence awareness
  - Layer normalization
  - GELU activations

**Reinforcement Learning Agent (`rl_agent.py`):**
- Algorithm: Proximal Policy Optimization (PPO)
- Purpose: Learn optimal trading actions through trial and error
- Features:
  - Policy and value networks
  - Reward shaping for risk-adjusted returns
  - Experience replay
  - Advantage estimation

#### 3. Ensemble Strategy (`ensemble_strategy.py`)

**Purpose:** Combine predictions from multiple AI models for robust decisions

**Methodology:**
- Weighted voting system
- Confidence-based weighting
- Dynamic weight adjustment based on performance
- Signal aggregation (buy/sell/hold)

**Decision Process:**
1. Collect predictions from all models
2. Calculate confidence scores
3. Weight predictions based on historical performance
4. Aggregate to final signal
5. Apply confidence threshold

#### 4. Risk Management (`risk_manager.py`)

**Key Features:**
- Dynamic position sizing based on:
  - Account balance
  - Risk per trade (default 2%)
  - Stop-loss distance
  - Model confidence
- Stop-loss and take-profit automation
- Maximum drawdown protection (default 15%)
- Position tracking and validation

**Position Sizing Formula:**
```
Position Size = (Capital × Risk Per Trade) / (Price × Stop Loss %)
Adjusted Size = Base Size × Confidence × Max Position Limit
```

#### 5. Gate.io Client (`gateio_client.py`)

**Capabilities:**
- Authenticated API requests
- Order placement (market and limit)
- Real-time market data
- Account balance management
- Order management

**Security:**
- HMAC-SHA512 signature authentication
- Secure credential handling
- Request signing

#### 6. Backtesting Framework (`backtest.py`)

**Features:**
- Historical data simulation
- Realistic trade execution
- Commission and slippage modeling
- Comprehensive performance metrics

#### 7. Main Bot (`main.py`)

**Orchestration:**
- Load trained models
- Schedule trading iterations (every 5 minutes)
- Process multiple trading pairs
- Execute trades based on signals
- Manage positions and risk
- Log all activities

### Trading Flow

```
1. Fetch Market Data
   ↓
2. Preprocess & Calculate Features
   ↓
3. Generate Predictions (LSTM + Transformer + RL)
   ↓
4. Ensemble Aggregation
   ↓
5. Risk Validation
   ↓
6. Execute Trade (if validated)
   ↓
7. Monitor Positions (stop-loss/take-profit)
   ↓
8. Update Performance Metrics
   ↓
9. Repeat (every 5 minutes)
```

### Model Training Flow

```
1. Fetch Historical Data (3000+ candles)
   ↓
2. Calculate Technical Indicators
   ↓
3. Create Sequences (lookback window)
   ↓
4. Split Train/Validation (80/20)
   ↓
5. Train Models (LSTM, Transformer)
   ↓
6. Validate Performance
   ↓
7. Save Models
```

### Key Design Decisions

1. **Ensemble Approach**: Combining multiple models reduces overfitting and improves robustness
2. **Risk-First Design**: Risk management is built into every trading decision
3. **Confidence-Based Sizing**: Position size scales with model confidence
4. **Modular Architecture**: Each component is independently testable
5. **Extensive Logging**: All decisions are logged for analysis

## Risk Management

### Position Sizing

Position size is calculated dynamically based on:
- Account balance
- Risk per trade percentage (default: 2%)
- Stop-loss distance
- Model confidence score

This ensures:
- Consistent risk exposure across trades
- Smaller positions when confidence is low
- Maximum position limits are respected

### Stop-Loss & Take-Profit

- **Stop-Loss**: Automatically set when opening a position (default: 2% below entry)
- **Take-Profit**: Automatically set when opening a position (default: 4% above entry)
- **Trailing**: Positions are monitored and closed automatically

### Drawdown Protection

- **Maximum Drawdown**: Trading halts if drawdown exceeds threshold (default: 15%)
- **Protection**: Prevents catastrophic losses during adverse market conditions
- **Recovery**: Manual intervention required to resume after maximum drawdown

### Risk Parameters

Configure in `.env`:
```env
RISK_PER_TRADE=0.02          # Risk 2% per trade
MAX_POSITION_SIZE=0.1        # Max 10% per position
STOP_LOSS_PCT=0.02           # 2% stop loss
TAKE_PROFIT_PCT=0.04         # 4% take profit
MAX_DRAWDOWN_PCT=0.15        # 15% max drawdown
```

## Monitoring and Logging

### Log Files

All activities are logged to:
- `logs/trading_bot_YYYY-MM-DD.log` - Main trading log
- `logs/dry_run_results_YYYYMMDD_HHMMSS.json` - Dry-run results

### What's Logged

- Trade executions (open/close)
- Signal generation and confidence scores
- Position management
- Risk checks and validations
- Performance metrics
- Errors and warnings

### Viewing Logs

**Using Makefile:**
```bash
make logs              # List recent log files
```

**Manual:**
```bash
tail -f logs/trading_bot_*.log    # Follow live log
cat logs/trading_bot_*.log        # View complete log
```

### Monitoring Best Practices

1. **Monitor Initially**: Watch the bot closely for the first few days
2. **Check Logs Daily**: Review daily logs for patterns
3. **Track Performance**: Compare actual vs. expected results
4. **Watch for Errors**: Monitor error logs for issues
5. **Review Trades**: Analyze trade decisions and outcomes

## Troubleshooting

### Common Issues

#### 1. "API credentials not configured"

**Solution:**
- Check your `.env` file exists
- Verify `GATE_API_KEY` and `GATE_API_SECRET` are set
- Run `make check-env` to verify configuration

#### 2. "No data fetched"

**Possible Causes:**
- No internet connection
- API credentials don't have read permissions
- Incorrect trading pair format

**Solution:**
- Check internet connection
- Verify API credentials have appropriate permissions
- Use underscore format: `BTC_USDT` (not `BTC/USDT`)

#### 3. "Model file not found"

**Solution:**
- Run `make train` to train models first
- Check `MODEL_PATH` in `.env` configuration
- Verify model files exist in `./models/` directory

#### 4. Low Performance / Negative Returns

**Possible Causes:**
- Market conditions not suitable
- Models need retraining
- Risk parameters too aggressive
- Overfitting to training data

**Solution:**
- Retrain models with more recent data
- Adjust risk parameters (reduce risk per trade)
- Try different trading pairs or timeframes
- Review and optimize technical indicators
- Extend backtest/dry-run periods

#### 5. "Maximum drawdown exceeded"

**Solution:**
- This is a safety feature - trading has been halted
- Review what caused the drawdown
- Adjust strategy or risk parameters
- Manually reset if needed (modify code or restart with new capital)

#### 6. Dry-Run Issues

**"No trades executed":**
- Check confidence threshold (`MIN_CONFIDENCE_THRESHOLD` in `.env`)
- Verify risk parameters allow trades
- Check if market conditions are suitable

**"Models not found":**
- Run `make train` first
- Check `MODEL_PATH` configuration

### Getting Help

1. Check logs for detailed error messages
2. Review configuration in `.env`
3. Verify API credentials and permissions
4. Test with backtest/dry-run first
5. Start with small amounts if issues persist

## Examples

### Example 1: Complete Workflow

```bash
# 1. Setup
make setup
# Edit .env with API credentials
make install

# 2. Train
make train

# 3. Test
make backtest       # Historical validation
make dry-run        # Real-time simulation

# 4. Review results
cat logs/dry_run_results_*.json

# 5. Go live (when satisfied)
make run
```

### Example 2: Quick Test

```bash
# Train and test quickly
make train
make dry-run dry-run-days=3    # Quick 3-day test
```

### Example 3: Configuration Change

```bash
# Edit configuration
nano .env
# Change RISK_PER_TRADE=0.02 to RISK_PER_TRADE=0.01

# Test with new settings
make dry-run

# No code changes needed!
```

### Example 4: Multiple Trading Pairs

```env
# In .env file
TRADING_PAIRS=BTC_USDT,ETH_USDT,BNB_USDT,SOL_USDT
```

Train and test for all pairs:
```bash
make train        # Trains models for all pairs
make dry-run      # Tests all pairs
```

## Makefile Commands

Quick reference for all available commands:

### Setup
```bash
make install      # Install dependencies
make setup        # Create .env file from template
make check-env    # Verify .env configuration
make dev-setup    # Complete setup (install + configure)
```

### Training & Testing
```bash
make train        # Train all models
make backtest     # Run historical backtest
make dry-run      # Run dry-run (7 days)
make dry-run dry-run-days=30  # Custom duration
```

### Trading
```bash
make run          # Start live trading (with confirmation)
```

### Utilities
```bash
make status       # Check bot configuration
make logs         # View recent logs
make clean        # Clean temporary files
make test         # Run basic tests
make show-config  # Show current configuration
make help         # Show all commands
```

## Important Disclaimers

### Risk Warning ⚠️

**This bot is for educational purposes only.**

- **Cryptocurrency trading involves substantial risk of loss**
- **You can lose all of your invested capital**
- **Past performance does not guarantee future results**
- **Only trade with funds you can afford to lose**

### No Guarantees

- **10-20% monthly returns are aspirational goals, not guarantees**
- Market conditions vary significantly
- All trading strategies have periods of losses
- AI models can make incorrect predictions

### Testing Requirements

- **Always test thoroughly** with backtesting and dry-run before live trading
- **Start with small amounts** when going live
- **Monitor closely** initially
- **Understand the risks** before trading

### Market Conditions

- Cryptocurrency markets are highly volatile
- Prices can change rapidly
- Network issues can affect execution
- API failures can occur

### Limitation of Liability

This software is provided "as-is" for educational purposes. The authors are not responsible for:
- Financial losses incurred through use of this bot
- Incorrect predictions or trading decisions
- Technical failures or bugs
- API outages or network issues
- Market conditions or volatility

### Your Responsibility

- Review and understand all code before using
- Test extensively in simulation mode
- Monitor bot performance regularly
- Adjust parameters based on your risk tolerance
- Comply with all applicable laws and regulations
- Ensure API keys are kept secure

---

## License

This software is provided for educational purposes. Use at your own risk.

---

**Remember**: Trading involves risk. Never invest more than you can afford to lose.

**Always test thoroughly before using real funds.**
