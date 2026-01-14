"""
Data Validator - Validates price data before trading decisions.
Ensures data freshness and validity.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pytz

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DataValidator:
    """Validates price data for trading"""
    
    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize data validator.
        
        Args:
            config: Data validation configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.max_data_age_minutes = config.get('max_data_age_minutes', 15)
        self.max_daily_change_percent = config.get('max_daily_change_percent', 50)
    
    def validate_price_data(self, df: Any, symbol: str) -> Tuple[bool, str]:
        """
        Validate price data for trading.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not HAS_PANDAS:
            return True, "Pandas not available, skipping validation"
        
        # Check if data exists
        if df is None or (hasattr(df, 'empty') and df.empty):
            return False, f"{symbol}: No data received"
        
        # Check for required columns
        required = ['Open', 'High', 'Low', 'Close']
        if hasattr(df, 'columns'):
            missing = [col for col in required if col not in df.columns]
            if missing:
                return False, f"{symbol}: Missing columns: {missing}"
        
        # Check data freshness
        try:
            if hasattr(df, 'index') and len(df.index) > 0:
                last_timestamp = df.index[-1]

                # Handle timezone
                if hasattr(last_timestamp, 'tz') and last_timestamp.tz is None:
                    last_timestamp = last_timestamp.tz_localize('UTC')
                elif hasattr(last_timestamp, 'tzinfo') and last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=pytz.UTC)
                                       
                now = datetime.now(pytz.UTC)
                if hasattr(last_timestamp, 'to_pydatetime'):
                    last_timestamp = last_timestamp.to_pydatetime()
                    
                # Ensure both are timezone-aware for comparison
                if hasattr(last_timestamp, 'tzinfo') and last_timestamp.tzinfo:
                    age = now - last_timestamp
                    max_age = timedelta(minutes=self.max_data_age_minutes)
                        
                    if age > max_age:
                        age_minutes = age.total_seconds() / 60
                        return False, f"{symbol}: Data is {age_minutes:.0f}min old (max: {self.max_data_age_minutes}min)"
        except Exception as e:
            self.logger.warning(f"Could not check data freshness for {symbol}: {e}")
        
        # Check for NaN values in OHLC
        try:
            if hasattr(df, 'isna'):
                ohlc_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
                nan_counts = df[ohlc_cols].isna().sum()
                if nan_counts.any():
                    return False, f"{symbol}: NaN values in {nan_counts[nan_counts > 0].to_dict()}"
        except Exception as e:
            self.logger.warning(f"Could not check NaN values for {symbol}: {e}")
        
        # Check for suspicious price movements
        try:
            if hasattr(df, 'pct_change') and len(df) >= 2:
                daily_change = abs(df['Close'].pct_change().iloc[-1])
                if daily_change > self.max_daily_change_percent / 100:
                    self.logger.warning(
                        f"{symbol}: Large price move: {daily_change*100:.1f}%"
                    )
                    # Don't reject, just warn - could be valid news
        except Exception as e:
            self.logger.warning(f"Could not check price change for {symbol}: {e}")
        
        # Check for zero/negative prices
        try:
            if 'Close' in df.columns and (df['Close'] <= 0).any():
                return False, f"{symbol}: Invalid price values (zero or negative)"
        except Exception:
            pass
        
        # Check for OHLC consistency
        try:
            if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
                issues = []
                warnings = []
                bad_dates = []
                
                # CRITICAL: Check High >= Low (should never happen, indicates bad data)
                high_low_issue = df['High'] < df['Low']
                if high_low_issue.any():
                    bad_rows = df[high_low_issue]
                    if hasattr(bad_rows.index[0], 'strftime'):
                        dates = bad_rows.index.strftime('%Y-%m-%d').tolist()
                    else:
                        dates = [str(d) for d in bad_rows.index]
                    issues.append(f"High<Low on {high_low_issue.sum()} date(s)")
                    bad_dates.extend(dates[:3])
                
                # MINOR: Check High >= Close (common yfinance issue, auto-fixed by data cleaning)
                high_close_issue = df['High'] < df['Close']
                if high_close_issue.any():
                    bad_rows = df[high_close_issue]
                    if hasattr(bad_rows.index[0], 'strftime'):
                        dates = bad_rows.index.strftime('%Y-%m-%d').tolist()
                    else:
                        dates = [str(d) for d in bad_rows.index]
                    warnings.append(f"High<Close on {high_close_issue.sum()} date(s)")
                    # Log warning but don't fail
                    self.logger.debug(f"{symbol}: Minor OHLC inconsistency (High<Close) on {dates[0] if dates else 'unknown'} - should be auto-fixed")
                
                # MINOR: Check High >= Open (common yfinance issue, auto-fixed by data cleaning)
                high_open_issue = df['High'] < df['Open']
                if high_open_issue.any():
                    bad_rows = df[high_open_issue]
                    if hasattr(bad_rows.index[0], 'strftime'):
                        dates = bad_rows.index.strftime('%Y-%m-%d').tolist()
                    else:
                        dates = [str(d) for d in bad_rows.index]
                    warnings.append(f"High<Open on {high_open_issue.sum()} date(s)")
                    # Log warning but don't fail
                    self.logger.debug(f"{symbol}: Minor OHLC inconsistency (High<Open) on {dates[0] if dates else 'unknown'} - should be auto-fixed")
                
                # Only fail on critical errors (High < Low)
                if issues:
                    unique_dates = ', '.join(sorted(set(bad_dates))[:3])
                    return False, f"{symbol}: Critical OHLC error - {'; '.join(issues)} on {unique_dates}"
                
                # Log warnings for minor issues (these should be auto-fixed by data cleaning)
                if warnings:
                    self.logger.debug(f"{symbol}: Minor OHLC inconsistencies detected (should be auto-fixed): {'; '.join(warnings)}")
        except Exception as e:
            self.logger.warning(f"Could not check OHLC consistency for {symbol}: {e}")
        
        return True, "OK"
    
    def validate_price(self, price_data: Dict[str, Any], symbol: str) -> Tuple[bool, str]:
        """
        Validate a single price data point.
        
        Args:
            price_data: Dictionary with price data
            symbol: Stock symbol
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if price_data is None:
            return False, f"{symbol}: No price data"
        
        # Check required fields
        required_fields = ['close']
        for field in required_fields:
            if field not in price_data and 'Close' not in price_data:
                return False, f"{symbol}: Missing {field} in price data"
        
        # Get close price
        close = price_data.get('close', price_data.get('Close', 0))
        
        # Check for valid price
        if close <= 0:
            return False, f"{symbol}: Invalid close price: {close}"
        
        # Check timestamp if available
        if 'timestamp' in price_data:
            try:
                timestamp = price_data['timestamp']
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=pytz.UTC)
                    
                    age = datetime.now(pytz.UTC) - timestamp
                    max_age = timedelta(minutes=self.max_data_age_minutes)
                    
                    if age > max_age:
                        age_minutes = age.total_seconds() / 60
                        return False, f"{symbol}: Price is {age_minutes:.0f}min old"
            except Exception:
                pass
        
        return True, "OK"
    
    def validate_all(self, data_dict: Dict[str, Any]) -> Dict[str, Tuple[bool, str]]:
        """
        Validate data for all symbols.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames or price data
            
        Returns:
            Dictionary mapping symbols to (is_valid, reason)
        """
        results = {}
        for symbol, data in data_dict.items():
            if HAS_PANDAS and hasattr(data, 'columns'):
                results[symbol] = self.validate_price_data(data, symbol)
            else:
                results[symbol] = self.validate_price(data, symbol)
        return results
    
    def filter_valid(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter to only valid data.
        
        Args:
            data_dict: Dictionary mapping symbols to data
            
        Returns:
            Dictionary with only valid data
        """
        valid_data = {}
        validation_results = self.validate_all(data_dict)
        
        for symbol, (is_valid, reason) in validation_results.items():
            if is_valid:
                valid_data[symbol] = data_dict[symbol]
            else:
                self.logger.warning(f"Filtered out {symbol}: {reason}")
        
        return valid_data


