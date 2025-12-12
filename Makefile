.PHONY: help install setup train backtest dry-run run clean test check-env

# Default target
help:
	@echo "=========================================="
	@echo "AI Trading Bot - Makefile Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make setup            - Copy env.template to .env"
	@echo "  make check-env        - Verify .env file configuration"
	@echo ""
	@echo "Training Commands:"
	@echo "  make train            - Train all models for configured trading pairs"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make backtest         - Run backtest on historical data"
	@echo "  make dry-run          - Run dry-run simulation (default: 7 days)"
	@echo "  make dry-run-days=30  - Run dry-run for N days (example: 30 days)"
	@echo ""
	@echo "Trading Commands:"
	@echo "  make run              - Start live trading bot (USE WITH CAUTION!)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean            - Remove cached files, logs, and temp data"
	@echo "  make test             - Run basic functionality tests"
	@echo "  make logs             - View recent logs"
	@echo "  make status           - Check bot status and configuration"
	@echo ""
	@echo "=========================================="

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Setup environment file
setup:
	@if [ ! -f .env ]; then \
		if [ -f env.template ]; then \
			cp env.template .env; \
			echo "✅ Created .env file from env.template"; \
		else \
			touch .env; \
			echo "✅ Created empty .env file"; \
		fi; \
		echo "⚠️  Please edit .env file with your API credentials"; \
	else \
		echo "⚠️  .env file already exists. Skipping..."; \
	fi

# Check environment configuration
check-env:
	@echo "Checking environment configuration..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@PYTHONPATH=. python -c "from dotenv import load_dotenv; import os; load_dotenv(); \
		api_key = os.getenv('GATE_API_KEY', ''); \
		api_secret = os.getenv('GATE_API_SECRET', ''); \
		pairs = os.getenv('TRADING_PAIRS', ''); \
		capital = os.getenv('INITIAL_CAPITAL', ''); \
		print('API Key:', '✅ Set' if api_key and api_key != 'your_api_key_here' else '❌ Not set'); \
		print('API Secret:', '✅ Set' if api_secret and api_secret != 'your_api_secret_here' else '❌ Not set'); \
		print('Trading Pairs:', pairs if pairs else '❌ Not set'); \
		print('Initial Capital:', capital if capital else '❌ Not set');"
	@echo "✅ Environment check complete"

# Train models
train: check-env
	@echo "=========================================="
	@echo "Training AI Models"
	@echo "=========================================="
	@echo "⚠️  This may take 10-30 minutes per trading pair..."
	@echo ""
	PYTHONPATH=. python -m src.train_models

# Run backtest
backtest: check-env
	@echo "=========================================="
	@echo "Running Backtest"
	@echo "=========================================="
	@echo ""
	PYTHONPATH=. python -m src.backtest

# Run dry-run simulation
dry-run: check-env
	@echo "=========================================="
	@echo "Starting Dry-Run Simulation"
	@echo "=========================================="
	@echo "⚠️  This simulates trading without placing real orders"
	@echo ""
	@if [ -z "$(dry-run-days)" ]; then \
		PYTHONPATH=. python -m src.dry_run 7; \
	else \
		echo "Running for $(dry-run-days) days..."; \
		PYTHONPATH=. python -m src.dry_run $(dry-run-days); \
	fi

# Run live trading bot
run: check-env
	@echo "=========================================="
	@echo "⚠️  WARNING: LIVE TRADING MODE"
	@echo "=========================================="
	@echo "This will start the bot with REAL trading enabled!"
	@echo "Make sure you have:"
	@echo "  1. Tested with backtesting and dry-run"
	@echo "  2. Reviewed all configuration settings"
	@echo "  3. Started with small amounts"
	@echo "  4. Understood the risks"
	@echo ""
	@read -p "Are you sure you want to continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Starting live trading bot..."; \
		PYTHONPATH=. python -m src.main; \
	else \
		echo "Cancelled."; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__ src/__pycache__ src/*/__pycache__
	@rm -rf .pytest_cache
	@rm -rf *.pyc */*.pyc */*/*.pyc src/*.pyc
	@rm -rf .mypy_cache
	@rm -rf *.egg-info
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# View recent logs
logs:
	@echo "Recent logs:"
	@if [ -d logs ]; then \
		ls -lt logs/*.log 2>/dev/null | head -5 || echo "No log files found"; \
	else \
		echo "No logs directory found"; \
	fi

# Check bot status and configuration
status: check-env
	@echo "=========================================="
	@echo "Bot Status and Configuration"
	@echo "=========================================="
	@PYTHONPATH=. python -c "from src.config import Config; \
		print('Trading Pairs:', ', '.join(Config.TRADING_PAIRS)); \
		print('Initial Capital:', f'\$${Config.INITIAL_CAPITAL:,.2f}'); \
		print('Risk Per Trade:', f'{Config.RISK_PER_TRADE*100:.1f}%'); \
		print('Primary Timeframe:', Config.PRIMARY_TIMEFRAME); \
		print('Model Path:', Config.MODEL_PATH); \
		print('Log Path:', Config.LOG_PATH);"
	@echo ""
	@if [ -d models ]; then \
		echo "Trained Models:"; \
		ls -1 models/*.pt 2>/dev/null | wc -l | xargs echo "  Found:" || echo "  None"; \
	else \
		echo "Models directory not found. Run 'make train' first."; \
	fi

# Basic functionality test
test:
	@echo "Running basic functionality tests..."
	@PYTHONPATH=. python -c "from src.config import Config; print('✅ Config loaded'); \
		from src.gateio_client import GateIOClient; print('✅ GateIO client imported'); \
		from src.data_pipeline import DataPipeline; print('✅ Data pipeline imported'); \
		from src.lstm_model import LSTMModel; from src.transformer_model import TransformerModel; print('✅ Models imported'); \
		print('✅ All imports successful');"

# Development shortcuts
dev-setup: install setup
	@echo "✅ Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Edit .env file with your API credentials"
	@echo "  2. Run 'make train' to train models"
	@echo "  3. Run 'make backtest' to test strategy"
	@echo "  4. Run 'make dry-run' to simulate trading"

# Quick backtest for one pair
backtest-quick: check-env
	@echo "Running quick backtest..."
	@PYTHONPATH=. python -c "from src.config import Config; \
		Config.TRADING_PAIRS = [Config.TRADING_PAIRS[0]]; \
		from src.backtest import train_models_and_backtest; \
		from src.gateio_client import GateIOClient; \
		client = GateIOClient(Config.GATE_API_KEY, Config.GATE_API_SECRET); \
		train_models_and_backtest(client, Config.TRADING_PAIRS[0]);"

# Show configuration values
show-config:
	@echo "Current Configuration:"
	@PYTHONPATH=. python -c "from src.config import Config; \
		import inspect; \
		members = inspect.getmembers(Config, lambda x: not inspect.isroutine(x)); \
		attrs = [m for m in members if not m[0].startswith('_') and not callable(m[1])]; \
		for name, value in attrs: \
			if not name.startswith('TECHNICAL') and not name.startswith('TIMEFRAMES'): \
				print(f'{name}: {value}');"

