import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PriceStatistics:
    """Calculate price statistics"""
    
    @staticmethod
    def calculate_basic_stats(prices: pd.Series) -> Dict:
        """Calculate basic price statistics"""
        if len(prices) < 2:
            return {}
        
        returns = prices.pct_change().dropna()
        
        return {
            'current_price': float(prices.iloc[-1]),
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
            'skewness': float(prices.skew()),
            'kurtosis': float(prices.kurtosis()),
            'var_95': float(np.percentile(prices, 5)),
            'var_99': float(np.percentile(prices, 1))
        }
    
    @staticmethod
    def calculate_rolling_stats(prices: pd.Series, window: int = 20) -> pd.DataFrame:
        """Calculate rolling statistics"""
        if len(prices) < window:
            return pd.DataFrame()
        
        df = pd.DataFrame({'price': prices})
        
        df['rolling_mean'] = prices.rolling(window=window).mean()
        df['rolling_std'] = prices.rolling(window=window).std()
        df['rolling_min'] = prices.rolling(window=window).min()
        df['rolling_max'] = prices.rolling(window=window).max()
        df['bollinger_upper'] = df['rolling_mean'] + 2 * df['rolling_std']
        df['bollinger_lower'] = df['rolling_mean'] - 2 * df['rolling_std']
        
        returns = prices.pct_change()
        df['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def calculate_volume_profile(ticks: List[Dict], price_bins: int = 50) -> Dict:
        """Calculate volume profile (Price * Volume distribution)"""
        if not ticks:
            return {}
        
        df = pd.DataFrame(ticks)
        if len(df) < 2:
            return {}
        
        # Create price bins
        min_price = df['price'].min()
        max_price = df['price'].max()
        price_range = max_price - min_price
        bin_size = price_range / price_bins
        
        bins = np.arange(min_price, max_price + bin_size, bin_size)
        df['price_bin'] = pd.cut(df['price'], bins=bins)
        
        # Calculate volume per bin
        volume_profile = df.groupby('price_bin')['size'].sum().to_dict()
        
        # Calculate Point of Control (POC)
        poc_bin = max(volume_profile, key=volume_profile.get)
        
        return {
            'volume_profile': volume_profile,
            'poc': {
                'price_range': str(poc_bin),
                'volume': float(volume_profile[poc_bin])
            },
            'total_volume': float(df['size'].sum()),
            'value_area': float(df['price'].dot(df['size']) / df['size'].sum())
        }
    
    @staticmethod
    def calculate_market_quality(prices: pd.Series, volumes: pd.Series) -> Dict:
        """Calculate market quality metrics"""
        if len(prices) < 10:
            return {}
        
        returns = prices.pct_change().dropna()
        log_returns = np.log(prices).diff().dropna()
        
        # Bid-ask spread estimate (simplified)
        spread_estimate = returns.abs().mean() * 10000  # In basis points
        
        # Amihud illiquidity ratio
        illiquidity = (returns.abs() / volumes.shift(1)).mean()
        
        # Price impact
        price_impact = (log_returns / np.sqrt(volumes.shift(1))).std()
        
        return {
            'spread_bps': float(spread_estimate),
            'illiquidity_ratio': float(illiquidity),
            'price_impact': float(price_impact),
            'efficiency_ratio': float(abs(log_returns).sum() / 
                                     np.sqrt((log_returns ** 2).sum() * len(log_returns)))
        }