"""
KIS API authentication with automatic token refresh.
Handles OAuth2 token management for Korea Investment & Securities API.
"""

from datetime import datetime, timedelta
import requests
import threading
from typing import Optional, Dict, Any
import logging


class KISAuth:
    """KIS API authentication with automatic token refresh"""
    
    # API URLs
    REAL_URL = "https://openapi.koreainvestment.com:9443"
    VIRTUAL_URL = "https://openapivts.koreainvestment.com:29443"
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize KIS authentication.
        
        Args:
            config: Dictionary containing app_key, app_secret, account_number, account_prod
            logger: Optional logger instance
        """
        self.app_key = config.get('app_key', '')
        self.app_secret = config.get('app_secret', '')
        self.account_number = config.get('account_number', '')
        self.account_prod = config.get('account_prod', '01')
        self.logger = logger or logging.getLogger(__name__)
        
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.token_lock = threading.Lock()
        
        # Token refresh settings
        self.token_lifetime_hours = 23  # Refresh 1 hour before 24h expiry
        self.base_url = self.REAL_URL
    
    def set_virtual_mode(self, is_virtual: bool = True) -> None:
        """Switch between real and virtual (paper) trading URLs"""
        self.base_url = self.VIRTUAL_URL if is_virtual else self.REAL_URL
        self.logger.info(f"KIS API mode: {'Virtual' if is_virtual else 'Real'}")
    
    def get_access_token(self) -> str:
        """Get valid access token, refreshing if needed"""
        with self.token_lock:
            if self._is_token_valid():
                return self.access_token
            
            self._refresh_token()
            return self.access_token
    
    def _is_token_valid(self) -> bool:
        """Check if current token is still valid"""
        if self.access_token is None or self.token_expires_at is None:
            return False
        
        # Refresh 1 hour before expiry
        buffer = timedelta(hours=1)
        return datetime.now() < (self.token_expires_at - buffer)
    
    def _refresh_token(self) -> None:
        """Get new access token from KIS API"""
        self.logger.info("Refreshing KIS API access token...")
        
        if not self.app_key or not self.app_secret:
            raise ValueError("KIS API credentials not configured")
        
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            response = requests.post(url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'access_token' not in data:
                error_msg = data.get('error_description', 'Unknown error')
                raise ValueError(f"Token request failed: {error_msg}")
            
            self.access_token = data['access_token']
            
            # Token valid for 24 hours from issue
            expires_in = int(data.get('expires_in', 86400))
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            self.logger.info(f"Token refreshed. Expires at: {self.token_expires_at}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to refresh token: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during token refresh: {e}")
            raise
    
    def get_headers(self, tr_id: str = "") -> Dict[str, str]:
        """
        Get headers with valid authorization for API requests.
        
        Args:
            tr_id: Transaction ID for the specific API call
            
        Returns:
            Dictionary of headers
        """
        headers = {
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "content-type": "application/json; charset=utf-8"
        }
        
        if tr_id:
            headers["tr_id"] = tr_id
        
        return headers
    
    def get_account_info(self) -> Dict[str, str]:
        """Get account information for API requests"""
        return {
            "account_number": self.account_number,
            "account_prod": self.account_prod
        }
    
    def revoke_token(self) -> bool:
        """Revoke current access token"""
        if not self.access_token:
            return True
        
        url = f"{self.base_url}/oauth2/revokeP"
        headers = {"content-type": "application/json"}
        body = {
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "token": self.access_token
        }
        
        try:
            response = requests.post(url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            
            self.access_token = None
            self.token_expires_at = None
            self.logger.info("Token revoked successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke token: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        return self._is_token_valid()
    
    def get_token_expiry(self) -> Optional[datetime]:
        """Get token expiry time"""
        return self.token_expires_at
    
    def get_token_remaining_seconds(self) -> int:
        """Get remaining seconds until token expires"""
        if not self.token_expires_at:
            return 0
        
        remaining = (self.token_expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))


