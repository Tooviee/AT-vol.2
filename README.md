# AI-Powered Algorithmic Trading System (AT vol.2)

A production-ready, ML-enhanced algorithmic trading system for US stock markets that combines technical analysis with machine learning to generate and execute trading signals autonomously. The system features comprehensive risk management, real-time market monitoring, and both paper trading and live trading capabilities.

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
- **Web Dashboard**: FastAPI-based monitoring dashboard with real-time status
- **Structured Logging**: JSON logging with rotation and retention policies

## 🏗️ Architecture

The system is built with a modular architecture consisting of 20+ specialized components:

```
AT vol.2/
├── main.py                    # Main orchestrator and trading loop
├── modules/                   # 20+ core trading modules
│   ├── strategy.py           # Technical analysis engine
│   ├── risk_management.py    # Position sizing and risk controls
│   ├── order_manager.py      # Order state machine
│   ├── circuit_breaker.py   # Safety mechanisms
│   └── [15+ more modules]
├── ml/                       # Machine learning subsystem
│   ├── ml_strategy.py        # ML-enhanced strategy
│   ├── feature_extractor.py  # 22-feature extraction
│   ├── confidence_booster.py # LightGBM model
│   ├── trainer.py            # CLI training tool
│   └── ab_testing.py         # A/B testing framework
├── data_persistence/         # Database layer
│   ├── database.py           # SQLite with WAL
│   └── models.py             # SQLAlchemy ORM
├── dashboard/                # Web interface
│   └── app.py                # FastAPI application
└── tests/                    # Comprehensive test suite
```

## 🛠️ Technologies

- **Languages**: Python 3.11+
- **ML/AI**: LightGBM, scikit-learn, pandas, numpy
- **Web Framework**: FastAPI, uvicorn
- **Database**: SQLite with SQLAlchemy ORM
- **Trading APIs**: KIS API, yfinance
- **Testing**: pytest
- **Other**: Pydantic, Discord.py, exchange-calendars

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

### 4. Start Trading

```bash
# Paper trading mode (recommended for testing)
python main.py
```

### 5. Train ML Model (After Collecting Data)

```bash
# Analyze collected data
python -m ml.trainer analyze

# Train model (requires 100+ trades)
python -m ml.trainer train --min-samples 100 --lookback-days 90
```

### 6. Access Dashboard

Open `http://127.0.0.1:8000` in your browser (if dashboard is enabled)

## 📚 Documentation

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** - Comprehensive project overview and architecture details
- **[RESUME_DESCRIPTION.md](RESUME_DESCRIPTION.md)** - Resume-ready project descriptions
- **[TRADING_SYSTEM_PLAN.md](TRADING_SYSTEM_PLAN.md)** - Detailed system architecture and implementation plan

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

## 📝 Status

**Current Version**: v2 (ML-Enhanced Architecture)

**Status**: Production-ready with paper trading mode active, ML model training in progress

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Always test thoroughly in paper trading mode before considering live trading. The authors are not responsible for any financial losses.

## 📄 License

[Add your license here]

---

For detailed architecture documentation, see [TRADING_SYSTEM_PLAN.md](TRADING_SYSTEM_PLAN.md)
