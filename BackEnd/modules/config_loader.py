"""
Configuration loader with Pydantic validation.
Loads and validates trading configuration from YAML file.
"""

import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class TimezoneConfig(BaseModel):
    local: str = "Asia/Seoul"
    market: str = "America/New_York"


class DisplayConfig(BaseModel):
    currency_label: str = "krw"
    show_usd_equivalent: bool = True


class StrategyConfig(BaseModel):
    sma_short: int = 10
    sma_long: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    atr_period: int = 14

    @field_validator('sma_short')
    @classmethod
    def sma_short_less_than_long(cls, v, info):
        return v

    @field_validator('sma_long')
    @classmethod
    def validate_sma_long(cls, v, info):
        if 'sma_short' in info.data and v <= info.data['sma_short']:
            raise ValueError('sma_long must be greater than sma_short')
        return v


class RiskConfig(BaseModel):
    risk_per_trade_percent: float = 1.0
    max_position_size_percent: float = 10.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_total_exposure_percent: float = 80.0
    max_drawdown_percent: float = 15.0
    max_correlated_positions: int = 3


class CircuitBreakerConfig(BaseModel):
    max_consecutive_losses: int = 3
    max_daily_loss_percent: float = 5.0
    max_daily_loss_krw: float = 500000
    loss_type: str = "realized"  # "realized" | "unrealized" | "both"
    api_error_threshold: int = 5
    cooldown_minutes: int = 30

    @field_validator('loss_type')
    @classmethod
    def validate_loss_type(cls, v):
        valid_types = ['realized', 'unrealized', 'both']
        if v not in valid_types:
            raise ValueError(f'loss_type must be one of {valid_types}')
        return v


class OrderManagementConfig(BaseModel):
    order_timeout_seconds: int = 60
    stale_check_interval_seconds: int = 30
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5


class MarketHoursConfig(BaseModel):
    regular_open: str = "09:30"
    regular_close: str = "16:00"
    trade_premarket: bool = False
    trade_afterhours: bool = False
    premarket_start: str = "04:00"
    afterhours_end: str = "20:00"
    skip_holidays: bool = True
    early_close_buffer_minutes: int = 30


class RateLimitsConfig(BaseModel):
    kis_api_calls_per_second: float = 5.0
    yfinance_calls_per_second: float = 2.0
    max_retry_attempts: int = 3
    base_retry_delay_seconds: float = 1.0


class HealthMonitorConfig(BaseModel):
    heartbeat_interval_seconds: int = 60
    alert_after_missed_heartbeats: int = 3
    check_interval_seconds: int = 300


class NetworkConfig(BaseModel):
    check_urls: List[str] = Field(default_factory=lambda: [
        "https://www.google.com",
        "https://finance.yahoo.com"
    ])
    timeout_seconds: int = 5
    failure_threshold: int = 3


class NotificationsConfig(BaseModel):
    large_order_threshold_krw: float = 5000000
    confirmation_timeout_seconds: int = 60


class PaperTradingConfig(BaseModel):
    initial_balance_krw: float = 10000000
    simulate_slippage: bool = True
    slippage_percent: float = 0.15
    simulate_spread: bool = True
    spread_percent: float = 0.05
    simulate_latency_ms: int = 100
    simulate_partial_fills: bool = False
    partial_fill_probability: float = 0.1


class ExchangeRateConfig(BaseModel):
    update_interval_minutes: int = 60
    fallback_rate: float = 1350.0


class DataValidationConfig(BaseModel):
    max_data_age_minutes: int = 15
    max_daily_change_percent: float = 50.0


class ReconciliationConfig(BaseModel):
    run_on_startup: bool = True
    run_interval_minutes: int = 30
    auto_sync_to_broker: bool = False


class DatabaseConfig(BaseModel):
    path: str = "BackEnd/data/trading.db"
    busy_timeout_ms: int = 5000
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_path: str = "BackEnd/data/backups/"
    keep_backups: int = 7


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    rotate_size_mb: int = 10
    keep_days: int = 30


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    session_timeout_minutes: int = 30


class MLModelParams(BaseModel):
    """LightGBM model parameters"""
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 100


class MLConfig(BaseModel):
    """Machine Learning configuration"""
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix
    
    enabled: bool = True
    min_confidence: float = 0.6
    min_confidence_ta_only: float = 0.4  # Lower threshold when ML not trained
    ml_weight: float = 0.5
    min_training_samples: int = 100
    retrain_interval_days: int = 7
    model_path: str = "BackEnd/ml/models/confidence_model.pkl"
    data_path: str = "BackEnd/ml/training_data"
    ab_testing: bool = False
    ab_test_ratio: float = 0.5
    model_params: MLModelParams = Field(default_factory=MLModelParams)


class KISConfig(BaseModel):
    """KIS API configuration from environment variables"""
    app_key: str = ""
    app_secret: str = ""
    account_number: str = ""
    account_prod: str = "01"

    @classmethod
    def from_env(cls) -> "KISConfig":
        return cls(
            app_key=os.getenv("KIS_APP_KEY", ""),
            app_secret=os.getenv("KIS_APP_SECRET", ""),
            account_number=os.getenv("KIS_ACCOUNT_NUMBER", ""),
            account_prod=os.getenv("KIS_ACCOUNT_PROD", "01")
        )


class DiscordConfig(BaseModel):
    """Discord configuration from environment variables"""
    bot_token: str = ""
    channel_id: int = 0

    @classmethod
    def from_env(cls) -> "DiscordConfig":
        return cls(
            bot_token=os.getenv("DISCORD_BOT_TOKEN", ""),
            channel_id=int(os.getenv("DISCORD_CHANNEL_ID", "0"))
        )


class TradingConfig(BaseModel):
    """Main configuration model"""
    mode: str = "paper"
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    
    timezone: TimezoneConfig = Field(default_factory=TimezoneConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    order_management: OrderManagementConfig = Field(default_factory=OrderManagementConfig)
    market_hours: MarketHoursConfig = Field(default_factory=MarketHoursConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    health_monitor: HealthMonitorConfig = Field(default_factory=HealthMonitorConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    exchange_rate: ExchangeRateConfig = Field(default_factory=ExchangeRateConfig)
    data_validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    
    # API configs loaded from environment
    kis: KISConfig = Field(default_factory=KISConfig.from_env)
    discord: DiscordConfig = Field(default_factory=DiscordConfig.from_env)

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        valid_modes = ['paper', 'live']
        if v not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')
        return v


class ConfigLoader:
    """Loads and validates trading configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[TradingConfig] = None
    
    def _find_config_file(self) -> str:
        """Find config file in standard locations"""
        possible_paths = [
            "usa_stock_trading_config.yaml",
            "config.yaml",
            Path(__file__).parent.parent / "usa_stock_trading_config.yaml"]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        raise FileNotFoundError(
            "Configuration file not found. Please create usa_stock_trading_config.yaml"
        )
    
    def load(self) -> TradingConfig:
        """Load and validate configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Inject environment variables
        raw_config['kis'] = KISConfig.from_env().model_dump()
        raw_config['discord'] = DiscordConfig.from_env().model_dump()
        
        # Validate with Pydantic
        self.config = TradingConfig(**raw_config)
        
        return self.config
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate configuration and return errors if any"""
        errors = []
        
        if self.config is None:
            errors.append("Configuration not loaded")
            return False, errors
        
        # Check for required API credentials in live mode
        if self.config.mode == 'live':
            if not self.config.kis.app_key:
                errors.append("KIS_APP_KEY not set (required for live trading)")
            if not self.config.kis.app_secret:
                errors.append("KIS_APP_SECRET not set (required for live trading)")
            if not self.config.kis.account_number:
                errors.append("KIS_ACCOUNT_NUMBER not set (required for live trading)")
        
        # Check symbols
        if not self.config.symbols:
            errors.append("No trading symbols configured")
        
        # Validate strategy parameters
        if self.config.strategy.sma_short >= self.config.strategy.sma_long:
            errors.append("SMA short period must be less than long period")
        
        return len(errors) == 0, errors
    
    def get_config(self) -> TradingConfig:
        """Get loaded configuration"""
        if self.config is None:
            self.load()
        return self.config


def load_config(config_path: Optional[str] = None) -> TradingConfig:
    """Convenience function to load configuration"""
    loader = ConfigLoader(config_path)
    config = loader.load()
    
    is_valid, errors = loader.validate()
    if not is_valid:
        raise ValueError(f"Configuration validation failed: {errors}")
    
    return config


