"""
KIS API Manager - Handles all interactions with Korea Investment & Securities API.
Provides methods for getting prices, placing orders, and managing positions.
"""

import logging
import time
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import pandas as pd

from .kis_auth_custom import KISAuth


# Configure yfinance with proper session
def _create_yf_session() -> requests.Session:
    """Create a requests session with proper headers for Yahoo Finance"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    return session


# Global session for yfinance
_YF_SESSION = _create_yf_session()


class KISAPIManager:
    """Manages all KIS API interactions"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize KIS API Manager.
        
        Args:
            config: Configuration dictionary containing KIS credentials
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        
        # Initialize authentication
        kis_config = config.get('kis', {})
        self.auth = KISAuth(kis_config, self.logger)
        
        # Set mode based on config
        is_paper = config.get('mode', 'paper') == 'paper'
        self.auth.set_virtual_mode(is_paper)
        
        self.base_url = self.auth.base_url
    
    def _make_request(self, method: str, endpoint: str, 
                      tr_id: str, params: Optional[Dict] = None,
                      data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated request to KIS API.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            tr_id: Transaction ID
            params: Query parameters
            data: Request body
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.auth.get_headers(tr_id)
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_price_yfinance(self, symbol: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Get current price using yfinance with retry logic.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with price data or None
        """
        import yfinance as yf
        
        for attempt in range(max_retries):
            try:
                # Use Ticker.history() which returns cleaner column structure
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1d')
                
                if hist is None or hist.empty:
                    # Add delay before retry
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.debug(f"No data for {symbol}, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    self.logger.warning(f"No data returned for {symbol} after {max_retries} attempts")
                    return None
                
                latest = hist.iloc[-1]
                
                # Handle potential Series/scalar values
                def get_val(val):
                    if hasattr(val, 'iloc'):
                        return val.iloc[0]
                    return val
                
                return {
                    'symbol': symbol,
                    'open': float(get_val(latest['Open'])),
                    'high': float(get_val(latest['High'])),
                    'low': float(get_val(latest['Low'])),
                    'close': float(get_val(latest['Close'])),
                    'volume': int(get_val(latest['Volume'])) if pd.notna(get_val(latest['Volume'])) else 0,
                    'timestamp': hist.index[-1].to_pydatetime()
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.debug(f"Error fetching {symbol}: {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to get price for {symbol} after {max_retries} attempts: {e}")
        
        return None
    
    def get_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current stock price.
        Uses yfinance as the data source for US stocks.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with price data or None
        """
        # For US stocks, use yfinance
        return self.get_price_yfinance(symbol)
    
    def get_prices(self, symbols: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get current prices for multiple symbols.
        Uses individual fetches with small delays to avoid rate limiting.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to price data
        """
        prices = {}
        
        for i, symbol in enumerate(symbols):
            prices[symbol] = self.get_price(symbol)
            
            # Add small delay between requests to avoid rate limiting
            if i < len(symbols) - 1:
                time.sleep(0.3)
        
        return prices
    
    def get_historical_data(self, symbol: str, period: str = '1mo', 
                            interval: str = '1d', max_retries: int = 3) -> Optional[Any]:
        """
        Get historical price data with retry logic.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            max_retries: Maximum retry attempts
            
        Returns:
            DataFrame with historical data or None
        """
        import yfinance as yf
        
        for attempt in range(max_retries):
            try:
                # Use Ticker.history() which returns clean single-level columns
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if hist is None or hist.empty:
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.debug(f"No historical data for {symbol}, retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    self.logger.warning(f"No historical data for {symbol} after {max_retries} attempts")
                    return None
                
                # Clean OHLC data - fix common yfinance data quality issues
                hist = self._clean_ohlc_data(hist, symbol)
                
                return hist
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.debug(f"Error fetching {symbol} history: {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to get historical data for {symbol} after {max_retries} attempts: {e}")
        
        return None
    
    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """
        Get account balance information.
        
        Returns:
            Dictionary with balance information or None
        """
        try:
            account_info = self.auth.get_account_info()
            
            # For US stocks overseas trading
            endpoint = "/uapi/overseas-stock/v1/trading/inquire-balance"
            tr_id = "TTTS3012R" if self.config.get('mode') == 'live' else "VTTS3012R"
            
            params = {
                "CANO": account_info['account_number'][:8],
                "ACNT_PRDT_CD": account_info['account_number'][8:] or "01",
                "OVRS_EXCG_CD": "NASD",  # NASDAQ
                "TR_CRCY_CD": "USD",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            
            response = self._make_request('GET', endpoint, tr_id, params=params)
            
            if response.get('rt_cd') == '0':
                return {
                    'total_balance': float(response.get('output2', [{}])[0].get('tot_evlu_amt', 0)),
                    'available_cash': float(response.get('output2', [{}])[0].get('frcr_pchs_amt', 0)),
                    'currency': 'USD'
                }
            else:
                self.logger.error(f"Failed to get balance: {response.get('msg1')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbols to position data
        """
        try:
            account_info = self.auth.get_account_info()
            
            endpoint = "/uapi/overseas-stock/v1/trading/inquire-balance"
            tr_id = "TTTS3012R" if self.config.get('mode') == 'live' else "VTTS3012R"
            
            params = {
                "CANO": account_info['account_number'][:8],
                "ACNT_PRDT_CD": account_info['account_number'][8:] or "01",
                "OVRS_EXCG_CD": "NASD",
                "TR_CRCY_CD": "USD",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            
            response = self._make_request('GET', endpoint, tr_id, params=params)
            
            positions = {}
            if response.get('rt_cd') == '0':
                for item in response.get('output1', []):
                    symbol = item.get('ovrs_pdno', '')
                    if symbol and int(item.get('ovrs_cblc_qty', 0)) > 0:
                        positions[symbol] = {
                            'symbol': symbol,
                            'quantity': int(item.get('ovrs_cblc_qty', 0)),
                            'avg_price': float(item.get('pchs_avg_pric', 0)),
                            'current_price': float(item.get('now_pric2', 0)),
                            'unrealized_pnl': float(item.get('frcr_evlu_pfls_amt', 0))
                        }
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def place_order(self, symbol: str, side: str, quantity: int,
                    order_type: str = 'market', limit_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Place an order.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            
        Returns:
            Order result or None
        """
        try:
            account_info = self.auth.get_account_info()
            
            # Determine transaction ID based on side and mode
            is_live = self.config.get('mode') == 'live'
            if side.lower() == 'buy':
                tr_id = "TTTT1002U" if is_live else "VTTT1002U"
            else:
                tr_id = "TTTT1006U" if is_live else "VTTT1006U"
            
            endpoint = "/uapi/overseas-stock/v1/trading/order"
            
            # Order type code
            if order_type == 'market':
                ord_dvsn = "00"  # Market order
                price = "0"
            else:
                ord_dvsn = "00"  # Limit order
                price = str(limit_price)
            
            data = {
                "CANO": account_info['account_number'][:8],
                "ACNT_PRDT_CD": account_info['account_number'][8:] or "01",
                "OVRS_EXCG_CD": "NASD",
                "PDNO": symbol,
                "ORD_QTY": str(quantity),
                "OVRS_ORD_UNPR": price,
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": ord_dvsn
            }
            
            response = self._make_request('POST', endpoint, tr_id, data=data)
            
            if response.get('rt_cd') == '0':
                return {
                    'order_id': response.get('output', {}).get('ODNO', ''),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'order_type': order_type,
                    'status': 'submitted',
                    'timestamp': datetime.now()
                }
            else:
                self.logger.error(f"Order failed: {response.get('msg1')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str, quantity: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Stock symbol
            quantity: Original order quantity
            
        Returns:
            True if cancelled successfully
        """
        try:
            account_info = self.auth.get_account_info()
            
            is_live = self.config.get('mode') == 'live'
            tr_id = "TTTT1004U" if is_live else "VTTT1004U"
            
            endpoint = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
            
            data = {
                "CANO": account_info['account_number'][:8],
                "ACNT_PRDT_CD": account_info['account_number'][8:] or "01",
                "OVRS_EXCG_CD": "NASD",
                "PDNO": symbol,
                "ORGN_ODNO": order_id,
                "RVSE_CNCL_DVSN_CD": "02",  # Cancel
                "ORD_QTY": str(quantity),
                "OVRS_ORD_UNPR": "0",
                "ORD_SVR_DVSN_CD": "0"
            }
            
            response = self._make_request('POST', endpoint, tr_id, data=data)
            
            if response.get('rt_cd') == '0':
                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                self.logger.error(f"Cancel failed: {response.get('msg1')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status or None
        """
        try:
            account_info = self.auth.get_account_info()
            
            is_live = self.config.get('mode') == 'live'
            tr_id = "TTTS3035R" if is_live else "VTTS3035R"
            
            endpoint = "/uapi/overseas-stock/v1/trading/inquire-nccs"
            
            params = {
                "CANO": account_info['account_number'][:8],
                "ACNT_PRDT_CD": account_info['account_number'][8:] or "01",
                "OVRS_EXCG_CD": "NASD",
                "SORT_SQN": "DS",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            
            response = self._make_request('GET', endpoint, tr_id, params=params)
            
            if response.get('rt_cd') == '0':
                for order in response.get('output', []):
                    if order.get('odno') == order_id:
                        return {
                            'order_id': order_id,
                            'status': self._parse_order_status(order),
                            'filled_quantity': int(order.get('ft_ccld_qty', 0)),
                            'remaining_quantity': int(order.get('nccs_qty', 0))
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def _parse_order_status(self, order_data: Dict) -> str:
        """Parse order status from API response"""
        # This is simplified - actual implementation would need proper status mapping
        filled_qty = int(order_data.get('ft_ccld_qty', 0))
        total_qty = int(order_data.get('ft_ord_qty', 0))
        
        if filled_qty == total_qty:
            return 'filled'
        elif filled_qty > 0:
            return 'partial_fill'
        else:
            return 'submitted'
    
    def ping(self) -> bool:
        """Check API connectivity"""
        try:
            # Try to get token - this will verify connectivity
            self.auth.get_access_token()
            return True
        except Exception:
            return False
    
    def _clean_ohlc_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean OHLC data to fix common yfinance data quality issues.
        
        Fixes:
        - High < Close or High < Open (sets High = max(High, Close, Open))
        - Low > Close or Low > Open (sets Low = min(Low, Close, Open))
        - High < Low (critical error, sets High = max(High, Low, Close, Open), Low = min)
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Create a copy to avoid modifying original
        cleaned = df.copy()
        fixed_count = 0
        
        # Fix High < Close or High < Open
        high_issues = (cleaned['High'] < cleaned['Close']) | (cleaned['High'] < cleaned['Open'])
        if high_issues.any():
            # Set High to max of High, Close, Open
            cleaned.loc[high_issues, 'High'] = cleaned.loc[high_issues, ['High', 'Close', 'Open']].max(axis=1)
            fixed_count += high_issues.sum()
        
        # Fix Low > Close or Low > Open
        low_issues = (cleaned['Low'] > cleaned['Close']) | (cleaned['Low'] > cleaned['Open'])
        if low_issues.any():
            # Set Low to min of Low, Close, Open
            cleaned.loc[low_issues, 'Low'] = cleaned.loc[low_issues, ['Low', 'Close', 'Open']].min(axis=1)
            fixed_count += low_issues.sum()
        
        # Fix High < Low (critical - should never happen)
        critical_issues = cleaned['High'] < cleaned['Low']
        if critical_issues.any():
            # Set High = max(High, Low, Close, Open), Low = min
            cleaned.loc[critical_issues, 'High'] = cleaned.loc[critical_issues, ['High', 'Low', 'Close', 'Open']].max(axis=1)
            cleaned.loc[critical_issues, 'Low'] = cleaned.loc[critical_issues, ['High', 'Low', 'Close', 'Open']].min(axis=1)
            fixed_count += critical_issues.sum()
            self.logger.warning(f"{symbol}: Fixed {critical_issues.sum()} critical OHLC errors (High < Low)")
        
        if fixed_count > 0:
            self.logger.debug(f"{symbol}: Auto-fixed {fixed_count} OHLC data quality issues from yfinance")
        
        return cleaned
    
    def close(self) -> None:
        """Clean up resources"""
        self.auth.revoke_token()


