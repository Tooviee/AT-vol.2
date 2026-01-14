# BackEnd - Trading Engine & ML System

This directory contains the core trading engine, machine learning subsystem, and all backend modules for the AT vol.2 trading system. The backend handles signal generation, order execution, risk management, data persistence, and ML model training.

## 🚀 Key Features

### Core Trading Engine
- **Hybrid Signal Generation**: Combines technical analysis (SMA, MACD, RSI, ATR) with ML-powered confidence boosting using LightGBM
- **Dual Trading Modes**: Paper trading simulation with realistic market conditions and live trading via KIS API
- **Real-time Market Awareness**: Intelligent market hours detection with holiday calendar support and timezone management
- **Automated Order Management**: State machine-based order lifecycle management with timeout handling and retry logic

### Machine Learning Integration
- **ML-Enhanced Confidence Scoring**: 22-feature ML model that enhances technical analysis signals
- **Continuous Learning**: Automated training data collection and periodic model retraining
- **A/B Testing Framework**: Compare ML-enhanced vs. standard trading performance

### Risk Management & Safety
- **Multi-layered Risk Controls**: ATR-based position sizing, drawdown protection, and correlation limits
- **Circuit Breaker System**: Automatic trading halt on consecutive losses or daily loss thresholds
- **Health Monitoring**: Heartbeat system, network connectivity checks, and Discord notifications

### Infrastructure
- **Data Persistence**: SQLite with WAL mode, automated backups, comprehensive trade tracking
- **Position Persistence**: Automatic position loading on startup and periodic syncing to database
- **Real-time Price Updates**: Automatic position price updates every 60 seconds (via yfinance)
- **Web Dashboard**: Django-based monitoring dashboard with real-time position tracking
- **Structured Logging**: JSON logging with rotation and retention policies

## 🏗️ Backend Architecture

The backend is built with a modular architecture consisting of 20+ specialized components:

```
BackEnd/
├── main.py                    # Main orchestrator and trading loop
├── modules/                   # 20+ core trading modules
│   ├── strategy.py           # Technical analysis engine
│   ├── ml_strategy.py        # ML-enhanced strategy wrapper
│   ├── risk_management.py    # Position sizing and risk controls
│   ├── order_manager.py      # Order state machine
│   ├── circuit_breaker.py   # Safety mechanisms
│   ├── balance_tracker.py   # Position and cash tracking
│   ├── paper_trading.py    # Paper trading executor
│   ├── kis_api_manager.py  # KIS API integration
│   ├── data_validator.py   # Price data validation
│   ├── market_hours.py     # Market hours and calendar
│   ├── timezone_utils.py   # Timezone management
│   ├── exchange_rate.py    # USD/KRW rate tracking
│   ├── health_monitor.py   # System health monitoring
│   ├── network_monitor.py  # Network connectivity checks
│   ├── notifier.py         # Discord notifications
│   ├── startup_recovery.py # Startup state recovery
│   ├── position_reconciler.py # Position reconciliation
│   └── [more modules...]
├── ml/                       # Machine learning subsystem
│   ├── ml_strategy.py        # ML-enhanced strategy
│   ├── feature_extractor.py  # 22-feature extraction
│   ├── confidence_booster.py # LightGBM model
│   ├── training_data_manager.py # Training data storage
│   ├── trainer.py            # CLI training tool
│   └── ab_testing.py         # A/B testing framework
├── data_persistence/         # Database layer
│   ├── database.py           # SQLite with WAL mode
│   └── models.py             # SQLAlchemy ORM models
└── tests/                    # Comprehensive test suite
```

## 🛠️ Backend Technologies

- **Language**: Python 3.11+
- **ML/AI**: LightGBM, scikit-learn, pandas, numpy
- **Database**: SQLite with SQLAlchemy ORM (WAL mode)
- **Trading APIs**: KIS API (live trading), yfinance (market data)
- **Configuration**: Pydantic models for type-safe config
- **Logging**: JSON structured logging with rotation
- **Testing**: pytest
- **Other**: Discord.py (notifications), exchange-calendars (market hours)

## 📋 Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list

## 🚦 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Copy `.env.example` to `.env` and add your API credentials:
   - KIS API keys
   - Discord bot token and channel ID

2. Review and customize `usa_stock_trading_config.yaml`:
   - Set trading mode (`paper` or `live`)
   - Configure symbols to trade
   - Adjust risk parameters
   - Enable/configure ML features

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Start Trading System

```bash
# Paper trading mode (recommended for testing)
python main.py
```

The system will:
- Load configuration from `usa_stock_trading_config.yaml`
- Initialize all modules (strategy, risk management, order manager, etc.)
- Load existing positions from database
- Start the trading loop

### 5. Train ML Model (After Collecting Data)

```bash
# Analyze collected training data
python -m ml.trainer analyze

# Train model (requires 100+ completed trades)
python -m ml.trainer train --min-samples 100 --lookback-days 90
```

### 6. Monitor via Web Dashboard

The Django frontend (in `../FrontEnd/`) connects to the same database and provides real-time monitoring. See [FrontEnd/README.md](../FrontEnd/README.md) for setup.

## 💾 Data Persistence & Recovery

The system automatically handles data persistence:

- **Position Persistence**: All positions are saved to the database immediately when trades execute and synced every 60 seconds. On restart, positions are automatically loaded from the database.
- **Price Updates**: Position prices are automatically updated every 60 seconds from market data (yfinance). Note: yfinance free tier has a 15-20 minute delay.
- **Startup Recovery**: The system automatically loads all positions, reconciles orders, and restores state on startup.
- **Database Location**: `BackEnd/data/trading.db` (SQLite with WAL mode for reliability)

## 📚 Backend Documentation

- **[TRADING_SYSTEM_PLAN.md](TRADING_SYSTEM_PLAN.md)** - Detailed system architecture and implementation plan
- **[../README.md](../README.md)** - Project overview and quick start
- **[../FrontEnd/README.md](../FrontEnd/README.md)** - Frontend dashboard documentation

## ⚙️ Configuration

The system is configured via `usa_stock_trading_config.yaml`. Key sections include:

- **Trading Mode**: `paper` or `live`
- **Symbols**: List of stocks to trade
- **Strategy**: Technical indicator parameters (SMA, MACD, RSI, ATR)
- **Risk Management**: Position sizing, stop-loss, take-profit, circuit breakers
- **ML Settings**: Enable/disable ML, confidence thresholds, training parameters
- **Market Hours**: Trading windows, holiday handling
- **Notifications**: Discord integration settings

## 🔒 Safety Features

- **Circuit Breakers**: Automatic halt on consecutive losses or daily loss limits
- **Risk Limits**: Maximum position size, total exposure, and drawdown protection
- **Data Validation**: Price anomaly detection and freshness checks
- **Health Monitoring**: System heartbeat and network connectivity monitoring
- **Graceful Shutdown**: State preservation and recovery mechanisms
- **Position Persistence**: All positions are automatically saved to database and restored on restart

## 📊 ML Model Details

The ML subsystem uses a LightGBM model with 22 engineered features:

- **Price Momentum**: 1-day, 5-day, 20-day price changes
- **Moving Averages**: SMA ratios and trends
- **MACD**: Histogram, signal, and trend indicators
- **RSI**: Value, zone, and change indicators
- **Volatility**: ATR percentage and volatility metrics
- **Volume**: Volume ratios
- **Temporal**: Day of week, hour of day
- **Signal Context**: Base confidence from technical analysis

The model enhances (never overrides) technical analysis signals by adjusting confidence scores based on learned patterns from historical trade outcomes.

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test coverage includes:
- Strategy and signal generation
- Risk management calculations
- Order management state machine
- Circuit breaker logic
- ML feature extraction
- ML confidence boosting

## 🔧 Backend Components

### Core Modules

- **`main.py`**: Main orchestrator that coordinates all modules and runs the trading loop
- **`modules/strategy.py`**: Technical analysis engine (SMA, MACD, RSI, ATR)
- **`modules/ml_strategy.py`**: ML-enhanced strategy wrapper
- **`modules/risk_management.py`**: Position sizing, stop-loss, take-profit calculation
- **`modules/order_manager.py`**: Order state machine with timeout handling
- **`modules/balance_tracker.py`**: In-memory position and cash tracking
- **`modules/paper_trading.py`**: Paper trading executor with slippage/spread simulation
- **`modules/kis_api_manager.py`**: KIS API integration for live trading and market data
- **`modules/data_validator.py`**: Price data validation and freshness checks
- **`modules/market_hours.py`**: NYSE market hours with holiday calendar support
- **`modules/circuit_breaker.py`**: Safety mechanisms to halt trading on losses
- **`modules/health_monitor.py`**: System heartbeat and health checks
- **`modules/notifier.py`**: Discord notifications for trades and alerts

### ML Subsystem

- **`ml/ml_strategy.py`**: ML-enhanced strategy that wraps base strategy
- **`ml/feature_extractor.py`**: Extracts 22 features from price data
- **`ml/confidence_booster.py`**: LightGBM model for confidence adjustment
- **`ml/training_data_manager.py`**: Manages training data in JSONL format
- **`ml/trainer.py`**: CLI tool for training and analyzing ML models
- **`ml/ab_testing.py`**: A/B testing framework for ML vs. control

### Data Persistence

- **`data_persistence/database.py`**: SQLite database manager with WAL mode
- **`data_persistence/models.py`**: SQLAlchemy ORM models (Order, Trade, Position, DailyPnL)

## 📝 Status

**Current Version**: v2 (ML-Enhanced Architecture)

**Status**: Production-ready with paper trading mode active, ML model training in progress

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Always test thoroughly in paper trading mode before considering live trading. The authors are not responsible for any financial losses.

## 📄 License

[Add your license here]

---

## 🔗 Related Documentation

- **[../README.md](../README.md)** - Project overview
- **[TRADING_SYSTEM_PLAN.md](TRADING_SYSTEM_PLAN.md)** - Detailed architecture plan
- **[../FrontEnd/README.md](../FrontEnd/README.md)** - Frontend dashboard
