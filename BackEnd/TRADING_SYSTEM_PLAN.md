# AI-Powered Auto Trading System - Final Architecture

## Overview

Production-ready US stock trading system with **ML-powered signal enhancement**. Combines technical analysis (SMA, MACD, RSI, ATR) with machine learning (LightGBM) for intelligent confidence boosting.

**Key Features:**
- Hybrid TA + ML signal generation
- Paper trading with realistic simulation
- Real-time market hours awareness (holidays, early close)
- Circuit breaker safety mechanisms
- Discord notifications with trade confirmations
- Web dashboard for monitoring
- SQLite with WAL mode for data persistence
- A/B testing for ML validation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USA AUTO TRADING SYSTEM v7                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Config    │    │  KIS API    │    │  yfinance   │    │   Discord   │  │
│  │   Loader    │    │  Manager    │    │  (fallback) │    │     Bot     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │         │
│         ▼                  ▼                  ▼                  ▼         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MAIN ORCHESTRATOR                            │   │
│  │  • Trading loop (10s interval)                                      │   │
│  │  • Market hours check                                               │   │
│  │  • Symbol processing                                                │   │
│  │  • Graceful shutdown                                                │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│         ┌───────────────────────┼───────────────────────┐                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │  ML-Enhanced│         │    Risk     │         │   Order     │          │
│  │  Strategy   │◄───────►│  Manager    │◄───────►│  Manager    │          │
│  └──────┬──────┘         └─────────────┘         └──────┬──────┘          │
│         │                                               │                  │
│         ▼                                               ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ML SUBSYSTEM                                │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │   Feature     │  │  Confidence   │  │   Training    │           │   │
│  │  │  Extractor    │─►│   Booster     │─►│    Manager    │           │   │
│  │  │  (22 features)│  │  (LightGBM)   │  │  (JSONL data) │           │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SAFETY SYSTEMS                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Circuit  │ │ Network  │ │  Health  │ │  Data    │ │ Position │  │   │
│  │  │ Breaker  │ │ Monitor  │ │ Monitor  │ │ Validator│ │Reconciler│  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA PERSISTENCE (SQLite WAL)                  │   │
│  │     Trades │ Orders │ Positions │ Daily P&L │ ML Training Data      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Requirements

```
Python: 3.11+ (required for zoneinfo, performance)
OS: Windows 10/11, Linux, macOS
RAM: 2GB minimum
Disk: 1GB for database, logs, and ML models
```

---

## Project Structure

```
AT vol.2/
├── main.py                      # Main orchestrator
├── usa_stock_trading_config.yaml # Configuration
├── requirements.txt             # Dependencies (pinned)
├── .env                         # Secrets (not in git)
├── .gitignore
│
├── modules/                     # Core modules
│   ├── __init__.py
│   ├── config_loader.py         # Pydantic config validation
│   ├── kis_api_manager.py       # KIS API interface
│   ├── kis_auth_custom.py       # Token refresh logic
│   ├── strategy.py              # Base TA strategy
│   ├── risk_management.py       # Position sizing
│   ├── balance_tracker.py       # Account tracking
│   ├── order_manager.py         # Order state machine
│   ├── paper_trading.py         # Simulation executor
│   ├── timezone_utils.py        # KST/EST handling
│   ├── market_hours.py          # Trading windows
│   ├── circuit_breaker.py       # Safety stops
│   ├── health_monitor.py        # Heartbeat system
│   ├── network_monitor.py       # Connectivity check
│   ├── notifier.py              # Discord notifications
│   ├── exchange_rate.py         # USD/KRW rates
│   ├── data_validator.py        # Price validation
│   ├── rate_limiter.py          # API throttling
│   ├── position_reconciler.py   # Broker sync
│   ├── startup_recovery.py      # State recovery
│   ├── shutdown_handler.py      # Graceful exit
│   ├── logger.py                # JSON logging
│   └── backtester.py            # Historical testing
│
├── ml/                          # Machine Learning (NEW)
│   ├── __init__.py
│   ├── feature_extractor.py     # 22 features for ML
│   ├── confidence_booster.py    # LightGBM model
│   ├── ml_strategy.py           # ML-enhanced strategy
│   ├── training_data_manager.py # Data collection
│   ├── trainer.py               # CLI training tool
│   ├── ab_testing.py            # A/B test manager
│   ├── models/                  # Saved models
│   │   └── confidence_model.pkl
│   └── training_data/           # Training data
│       └── training_data.jsonl
│
├── data_persistence/            # Database layer
│   ├── __init__.py
│   ├── database.py              # SQLite + WAL
│   └── models.py                # SQLAlchemy models
│
├── dashboard/                   # Web UI
│   ├── __init__.py
│   ├── app.py                   # FastAPI app
│   └── templates/               # Jinja2 templates
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_strategy.py
│   ├── test_risk_management.py
│   ├── test_order_manager.py
│   ├── test_circuit_breaker.py
│   ├── test_ml_feature_extractor.py
│   └── test_ml_confidence_booster.py
│
├── data/                        # Runtime data
│   ├── trading.db               # SQLite database
│   └── backups/                 # DB backups
│
└── logs/                        # Log files
    └── usa_stock_trading.log
```

---

## ML Integration: Hybrid Confidence Boosting

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID SIGNAL FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Price Data ──► Technical Analysis ──► Base Signal (BUY/SELL)  │
│                      │                        │                 │
│                      │                        ▼                 │
│                      │              Base Confidence (0.65)      │
│                      │                        │                 │
│                      ▼                        │                 │
│              Feature Extraction               │                 │
│              ┌─────────────────┐              │                 │
│              │ • RSI value     │              │                 │
│              │ • MACD histogram│              │                 │
│              │ • Volatility    │              │                 │
│              │ • Volume ratio  │              │                 │
│              │ • Day of week   │              │                 │
│              │ • Hour of day   │              │                 │
│              │ • Price momentum│              │                 │
│              └────────┬────────┘              │                 │
│                       │                       │                 │
│                       ▼                       │                 │
│              ┌─────────────────┐              │                 │
│              │   ML Model      │              │                 │
│              │  (LightGBM)     │──────────────┼─► Final        │
│              │   P(success)    │   Weighted   │   Confidence   │
│              └─────────────────┘   Average    │   (0.0 - 1.0)  │
│                                               │                 │
│                                               ▼                 │
│                                     Execute if confidence       │
│                                     >= threshold (0.6)          │
└─────────────────────────────────────────────────────────────────┘
```

### ML Features (22 total)

| Category | Features |
|----------|----------|
| **Price Momentum** | `price_change_1d`, `price_change_5d`, `price_change_20d` |
| **Moving Averages** | `sma_ratio`, `price_to_sma_short`, `price_to_sma_long`, `sma_trend` |
| **MACD** | `macd`, `macd_signal`, `macd_histogram`, `macd_hist_change`, `macd_trend` |
| **RSI** | `rsi`, `rsi_zone`, `rsi_change` |
| **Volatility** | `atr_percent`, `volatility`, `volatility_change` |
| **Volume** | `volume_ratio` |
| **Time** | `day_of_week`, `hour_of_day` |
| **Signal Context** | `base_confidence` |

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| ML enhances, never overrides | TA generates signal; ML only adjusts confidence |
| Lightweight model | LightGBM (fast, small data requirements) |
| Conservative threshold | Only trade when confidence >= 0.6 |
| Fallback mode | Works without ML (uses base confidence) |
| Continuous learning | Data collected automatically, retrain weekly |

### Training Pipeline

```bash
# Phase 1: Data is collected automatically as you trade

# Phase 2: Analyze collected data
python -m ml.trainer analyze

# Phase 3: Train the model (need 100+ trades)
python -m ml.trainer train --min-samples 100 --lookback-days 90

# Phase 4: Export for Jupyter analysis
python -m ml.trainer export -o ml/analysis.csv
```

### A/B Testing

Enable to compare ML vs standard performance:

```yaml
ml:
  ab_testing: true
  ab_test_ratio: 0.5  # 50% ML, 50% control
```

---

## Configuration Reference

### Complete Config (v7)

```yaml
# usa_stock_trading_config.yaml

# ===== SYSTEM =====
python_version: "3.11+"

# ===== TRADING MODE =====
mode: "paper"  # "paper" | "live"

# ===== TIMEZONE =====
timezone:
  local: "Asia/Seoul"
  market: "America/New_York"

# ===== DISPLAY =====
display:
  currency_label: "krw"
  show_usd_equivalent: true

# ===== SYMBOLS =====
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - NVDA
  - AMZN

# ===== STRATEGY =====
strategy:
  sma_short: 10
  sma_long: 30
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  atr_period: 14

# ===== RISK =====
risk:
  risk_per_trade_percent: 1
  max_position_size_percent: 10
  stop_loss_atr_multiplier: 2.0
  take_profit_atr_multiplier: 3.0
  max_total_exposure_percent: 80
  max_drawdown_percent: 15
  max_correlated_positions: 3

# ===== CIRCUIT BREAKER =====
circuit_breaker:
  max_consecutive_losses: 3
  max_daily_loss_percent: 5.0
  max_daily_loss_krw: 500000
  loss_type: "realized"  # "realized" | "unrealized" | "both"
  api_error_threshold: 5
  cooldown_minutes: 30

# ===== ORDER MANAGEMENT =====
order_management:
  order_timeout_seconds: 60
  stale_check_interval_seconds: 30
  max_retry_attempts: 3
  retry_delay_seconds: 5

# ===== MARKET HOURS =====
market_hours:
  regular_open: "09:30"
  regular_close: "16:00"
  trade_premarket: false
  trade_afterhours: false
  premarket_start: "04:00"
  afterhours_end: "20:00"
  skip_holidays: true
  early_close_buffer_minutes: 30

# ===== RATE LIMITS =====
rate_limits:
  kis_api_calls_per_second: 5
  yfinance_calls_per_second: 2
  max_retry_attempts: 3
  base_retry_delay_seconds: 1.0

# ===== HEALTH MONITOR =====
health_monitor:
  heartbeat_interval_seconds: 60
  alert_after_missed_heartbeats: 3
  check_interval_seconds: 300

# ===== NETWORK =====
network:
  check_urls:
    - "https://www.google.com"
    - "https://finance.yahoo.com"
  timeout_seconds: 5
  failure_threshold: 3

# ===== NOTIFICATIONS =====
notifications:
  large_order_threshold_krw: 5000000
  confirmation_timeout_seconds: 60

# ===== PAPER TRADING =====
paper_trading:
  initial_balance_krw: 10000000
  simulate_slippage: true
  slippage_percent: 0.15
  simulate_spread: true
  spread_percent: 0.05
  simulate_latency_ms: 100
  simulate_partial_fills: false
  partial_fill_probability: 0.1

# ===== EXCHANGE RATE =====
exchange_rate:
  update_interval_minutes: 60
  fallback_rate: 1450.0

# ===== DATA VALIDATION =====
data_validation:
  max_data_age_minutes: 15
  max_daily_change_percent: 50

# ===== RECONCILIATION =====
reconciliation:
  run_on_startup: true
  run_interval_minutes: 30
  auto_sync_to_broker: false

# ===== DATABASE =====
database:
  path: "data/trading.db"
  busy_timeout_ms: 5000
  backup_enabled: true
  backup_interval_hours: 24
  backup_path: "data/backups/"
  keep_backups: 7

# ===== LOGGING =====
logging:
  level: INFO
  format: json
  rotate_size_mb: 10
  keep_days: 30

# ===== DASHBOARD =====
dashboard:
  host: "127.0.0.1"
  port: 8000
  session_timeout_minutes: 30

# ===== MACHINE LEARNING ===== (NEW)
ml:
  enabled: true
  min_confidence: 0.6
  ml_weight: 0.5          # 50% TA, 50% ML
  min_training_samples: 100
  retrain_interval_days: 7
  model_path: "ml/models/confidence_model.pkl"
  data_path: "ml/training_data"
  ab_testing: false
  ab_test_ratio: 0.5
  model_params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
```

---

## Dependencies

```
# requirements.txt - v7 (with ML)

# Core
pyyaml==6.0.1
python-dotenv==1.0.0
pydantic==2.5.3

# Data & Trading
yfinance==0.2.33
pandas==2.1.4
numpy==1.26.2
requests==2.31.0
pytz==2023.3

# Database
sqlalchemy==2.0.23

# Scheduling
apscheduler==3.10.4

# Market Hours
exchange-calendars==4.5.2

# Discord Bot
discord.py==2.3.2

# Dashboard
fastapi==0.109.0
uvicorn==0.25.0
jinja2==3.1.2
python-multipart==0.0.6

# Charts
plotly==5.18.0

# Machine Learning (NEW)
lightgbm==4.1.0
scikit-learn==1.3.2
joblib==1.3.2

# Testing
pytest==7.4.3
pytest-asyncio==0.23.2
```

---

## Implementation Status

### ✅ Completed Modules

| Module | Status | Notes |
|--------|--------|-------|
| `config_loader.py` | ✅ Complete | Pydantic validation, ML config |
| `kis_auth_custom.py` | ✅ Complete | Token refresh logic |
| `kis_api_manager.py` | ✅ Complete | API interface |
| `strategy.py` | ✅ Complete | SMA, MACD, RSI, ATR |
| `risk_management.py` | ✅ Complete | Position sizing |
| `balance_tracker.py` | ✅ Complete | Account tracking |
| `order_manager.py` | ✅ Complete | State machine, timeouts |
| `paper_trading.py` | ✅ Complete | Market price fetching |
| `timezone_utils.py` | ✅ Complete | exchange-calendars |
| `market_hours.py` | ✅ Complete | Trading windows |
| `circuit_breaker.py` | ✅ Complete | Loss type config |
| `health_monitor.py` | ✅ Complete | Heartbeat |
| `network_monitor.py` | ✅ Complete | Connectivity |
| `notifier.py` | ✅ Complete | Discord bot |
| `exchange_rate.py` | ✅ Complete | USD/KRW |
| `data_validator.py` | ✅ Complete | Price validation |
| `rate_limiter.py` | ✅ Complete | API throttling |
| `position_reconciler.py` | ✅ Complete | Broker sync |
| `startup_recovery.py` | ✅ Complete | State recovery |
| `shutdown_handler.py` | ✅ Complete | Windows SIGBREAK |
| `logger.py` | ✅ Complete | JSON logging |
| `database.py` | ✅ Complete | SQLite WAL |
| `models.py` | ✅ Complete | SQLAlchemy ORM |
| `main.py` | ✅ Complete | Orchestrator with ML |
| `backtester.py` | ✅ Complete | Historical testing |
| `dashboard/app.py` | ✅ Complete | FastAPI app |

### ✅ ML Modules (NEW)

| Module | Status | Notes |
|--------|--------|-------|
| `ml/feature_extractor.py` | ✅ Complete | 22 features |
| `ml/confidence_booster.py` | ✅ Complete | LightGBM model |
| `ml/ml_strategy.py` | ✅ Complete | Enhanced strategy |
| `ml/training_data_manager.py` | ✅ Complete | Data collection |
| `ml/trainer.py` | ✅ Complete | CLI tool |
| `ml/ab_testing.py` | ✅ Complete | A/B test manager |

### ✅ Tests

| Test File | Status |
|-----------|--------|
| `test_strategy.py` | ✅ Complete |
| `test_risk_management.py` | ✅ Complete |
| `test_order_manager.py` | ✅ Complete |
| `test_circuit_breaker.py` | ✅ Complete |
| `test_ml_feature_extractor.py` | ✅ Complete |
| `test_ml_confidence_booster.py` | ✅ Complete |

---

## Pre-Launch Checklist

### Environment Setup

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] `.env` file created with API credentials

### Configuration

- [ ] `usa_stock_trading_config.yaml` reviewed
- [ ] Trading mode set to `paper`
- [ ] Symbols configured
- [ ] Risk parameters reviewed

### External Services

- [ ] KIS API credentials verified
- [ ] KIS account approved for API trading
- [ ] Discord bot created and token obtained
- [ ] Discord channel ID configured

### Testing

- [ ] Run unit tests: `pytest tests/ -v`
- [ ] Paper trade for 1+ week
- [ ] Verify market hours detection (test on Friday/weekend)
- [ ] Test shutdown with Ctrl+C
- [ ] Verify database backups work
- [ ] Check ATR-based stop-loss calculations
- [ ] Test KIS token refresh (run for 24+ hours)

### ML Training (After Collecting Data)

- [ ] Collect 100+ trades in paper mode
- [ ] Run `python -m ml.trainer analyze`
- [ ] Train model: `python -m ml.trainer train`
- [ ] Optionally enable A/B testing
- [ ] Monitor ML vs control performance

### Go Live

- [ ] All tests passing
- [ ] Paper trading profitable for 2+ weeks
- [ ] ML model trained and validated
- [ ] Change mode to `live`
- [ ] Start with small position sizes
- [ ] Monitor closely for first week

---

## Quick Start

```bash
# 1. Clone and setup
cd "C:\Users\PRO\Desktop\AT vol.2"
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
# Edit .env with your API keys
# Review usa_stock_trading_config.yaml

# 4. Run tests
pytest tests/ -v

# 5. Start paper trading
python main.py

# 6. (Later) Train ML model
python -m ml.trainer train

# 7. View dashboard
# Open http://127.0.0.1:8000
```

---

## Support

For issues or questions:
1. Check logs in `logs/usa_stock_trading.log`
2. Review configuration in `usa_stock_trading_config.yaml`
3. Run tests to verify module health
4. Check Discord for notifications

---

*Last Updated: December 30, 2025*

