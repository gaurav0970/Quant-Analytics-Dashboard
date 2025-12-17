import pandas as pd
import numpy as np
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm  # ADD THIS LINE
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class StatisticalTests:
    """Statistical tests for time series analysis"""
    
    @staticmethod
    def adf_test(series: pd.Series, maxlag: int = None) -> Dict:
        """Augmented Dickey-Fuller test for stationarity"""
        if len(series) < 10:
            return {}
        
        try:
            result = adfuller(series.dropna(), maxlag=maxlag)
            
            return {
                'test_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'n_obs': int(result[3]),
                'stationary': result[1] < 0.05,
                'interpretation': 'Series is stationary' if result[1] < 0.05 
                                else 'Series is non-stationary'
            }
        except Exception as e:
            logger.error(f"ADF test error: {e}")
            return {}
    
    @staticmethod
    def cointegration_test(x: pd.Series, y: pd.Series, 
                          maxlag: int = None) -> Dict:
        """Cointegration test between two series"""
        if len(x) < 10 or len(y) < 10:
            return {}
        
        try:
            # Align series
            df = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(df) < 10:
                return {}
            
            result = coint(df['x'], df['y'], maxlag=maxlag)
            
            return {
                'test_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[2].items()},
                'cointegrated': result[1] < 0.05,
                'interpretation': 'Series are cointegrated' if result[1] < 0.05 
                                else 'Series are not cointegrated'
            }
        except Exception as e:
            logger.error(f"Cointegration test error: {e}")
            return {}
    
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, 
                                max_lags: int = 20) -> float:
        """Calculate Hurst exponent for time series"""
        if len(series) < 100:
            return 0.5
        
        lags = range(2, min(max_lags, len(series) // 2))
        tau = []
        
        for lag in lags:
            # Calculate R/S for this lag
            series_lag = np.array(series)
            n = len(series_lag)
            k = n // lag
            
            # Reshape into k segments of length lag
            segments = series_lag[:k * lag].reshape(k, lag)
            
            # Calculate mean of each segment
            means = np.mean(segments, axis=1)
            
            # Calculate deviations from mean
            deviations = segments - means[:, np.newaxis]
            
            # Calculate cumulative deviations
            cum_dev = np.cumsum(deviations, axis=1)
            
            # Calculate range for each segment
            R = np.max(cum_dev, axis=1) - np.min(cum_dev, axis=1)
            
            # Calculate standard deviation for each segment
            S = np.std(segments, axis=1, ddof=1)
            
            # Avoid division by zero
            S[S == 0] = 1e-10
            
            # Calculate R/S for each segment
            RS = R / S
            
            # Average R/S for this lag
            tau.append(np.mean(RS))
        
        # Linear regression of log(tau) vs log(lag)
        if len(tau) > 1:
            x = np.log(lags)
            y = np.log(tau)
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.5
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        if len(spread) < 10:
            return 0
        
        try:
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            aligned = pd.DataFrame({
                'spread_lag': spread_lag,
                'spread_diff': spread_diff
            }).dropna()
            
            if len(aligned) < 5:
                return 0
            
            # Regression: Δspread = α + β * spread_lag
            X = sm.add_constant(aligned['spread_lag'])
            model = sm.OLS(aligned['spread_diff'], X).fit()
            
            beta = model.params['spread_lag']
            if beta >= 0:
                return 0
            
            half_life = -np.log(2) / beta
            return float(half_life)
        except Exception as e:
            logger.error(f"Half-life calculation error: {e}")
            return 0