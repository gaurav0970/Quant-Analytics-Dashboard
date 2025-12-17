import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

class RegressionAnalytics:
    """Regression-based analytics for pairs trading"""
    
    @staticmethod
    def calculate_ols_hedge_ratio(x: pd.Series, y: pd.Series, 
                                 add_constant: bool = True) -> Dict:
        """Calculate OLS hedge ratio between two series"""
        if len(x) < 10 or len(y) < 10:
            return {}
        
        # Align series
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 10:
            return {}
        
        X = df['x'].values.reshape(-1, 1)
        y_vals = df['y'].values
        
        if add_constant:
            X = sm.add_constant(X)
            model = sm.OLS(y_vals, X).fit()
            intercept = model.params[0]
            hedge_ratio = model.params[1]
            residuals = model.resid
        else:
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y_vals)
            intercept = 0
            hedge_ratio = model.coef_[0]
            residuals = y_vals - model.predict(X)
        
        # Calculate statistics
        spread_mean = residuals.mean()
        spread_std = residuals.std()
        zscore = (residuals[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        return {
            'hedge_ratio': float(hedge_ratio),
            'intercept': float(intercept),
            'r_squared': float(model.score(X, y_vals) if not add_constant else model.rsquared),
            'residuals': residuals.tolist(),
            'spread_mean': float(spread_mean),
            'spread_std': float(spread_std),
            'current_zscore': float(zscore),
            'model_summary': str(model.summary()) if add_constant else None
        }
    
    @staticmethod
    def calculate_robust_regression(x: pd.Series, y: pd.Series, 
                                   method: str = 'huber') -> Dict:
        """Calculate robust regression (Huber or Theil-Sen)"""
        if len(x) < 10 or len(y) < 10:
            return {}
        
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 10:
            return {}
        
        X = df['x'].values.reshape(-1, 1)
        y_vals = df['y'].values
        
        if method == 'huber':
            model = HuberRegressor()
        elif method == 'theil_sen':
            # Simple implementation of Theil-Sen
            slopes = []
            n = len(X)
            for i in range(n):
                for j in range(i+1, n):
                    if X[j] != X[i]:
                        slopes.append((y_vals[j] - y_vals[i]) / (X[j] - X[i]))
            
            if slopes:
                hedge_ratio = np.median(slopes)
                intercept = np.median(y_vals - hedge_ratio * X.flatten())
                residuals = y_vals - (intercept + hedge_ratio * X.flatten())
                r_squared = 1 - np.var(residuals) / np.var(y_vals)
                
                return {
                    'hedge_ratio': float(hedge_ratio),
                    'intercept': float(intercept),
                    'r_squared': float(r_squared),
                    'residuals': residuals.tolist(),
                    'method': 'theil_sen'
                }
            else:
                return {}
        else:
            return {}
        
        model.fit(X, y_vals)
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_
        residuals = y_vals - model.predict(X)
        r_squared = model.score(X, y_vals)
        
        return {
            'hedge_ratio': float(hedge_ratio),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'residuals': residuals.tolist(),
            'method': method
        }
    
    @staticmethod
    def calculate_kalman_filter_hedge(x: pd.Series, y: pd.Series,
                                     delta: float = 1e-4) -> Dict:
        """Dynamic hedge ratio estimation using Kalman Filter"""
        if len(x) < 10 or len(y) < 10:
            return {}
        
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        n = len(df)
        
        if n < 10:
            return {}
        
        x_vals = df['x'].values
        y_vals = df['y'].values
        
        # Kalman Filter initialization
        theta = np.zeros((2, 1))  # [hedge_ratio, intercept]
        P = np.eye(2) * 1000  # Large initial uncertainty
        R = delta / (1 - delta) * np.eye(1)  # Measurement noise
        Q = np.eye(2) * 1e-4  # Process noise
        
        hedge_ratios = []
        intercepts = []
        residuals = []
        
        for i in range(n):
            # Measurement
            H = np.array([[x_vals[i], 1]])
            z = y_vals[i]
            
            # Prediction update
            y_pred = H @ theta
            residual = z - y_pred[0, 0]
            residuals.append(residual)
            
            # Kalman gain
            S = H @ P @ H.T + R
            K = P @ H.T / S[0, 0]
            
            # State update
            theta = theta + K * residual
            P = (np.eye(2) - K @ H) @ P
            
            # Store current estimate
            hedge_ratios.append(theta[0, 0])
            intercepts.append(theta[1, 0])
            
            # Time update (random walk)
            theta = theta  # No change for random walk
            P = P + Q
        
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals)
        zscore = (residuals[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        return {
            'current_hedge_ratio': float(hedge_ratios[-1]),
            'current_intercept': float(intercepts[-1]),
            'hedge_ratios': [float(h) for h in hedge_ratios],
            'intercepts': [float(i) for i in intercepts],
            'residuals': [float(r) for r in residuals],
            'spread_mean': float(spread_mean),
            'spread_std': float(spread_std),
            'current_zscore': float(zscore),
            'method': 'kalman_filter'
        }
    
    @staticmethod
    def calculate_rolling_correlation(x: pd.Series, y: pd.Series, 
                                     window: int = 20) -> pd.Series:
        """Calculate rolling correlation between two series"""
        if len(x) < window or len(y) < window:
            return pd.Series()
        
        aligned = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(aligned) < window:
            return pd.Series()
        
        return aligned['x'].rolling(window=window).corr(aligned['y'])