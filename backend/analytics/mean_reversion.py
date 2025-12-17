import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from .regression import RegressionAnalytics
from .statistical_tests import StatisticalTests

logger = logging.getLogger(__name__)

class MeanReversionAnalytics:
    """Mean reversion trading analytics"""
    
    def __init__(self, entry_zscore: float = 2.0, exit_zscore: float = 0.0):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.positions = {}
    
    def calculate_spread_zscore(self, x: pd.Series, y: pd.Series, 
                               hedge_ratio: float = None) -> Dict:
        """Calculate spread and z-score between two series"""
        if len(x) < 10 or len(y) < 10:
            return {}
        
        # Align series
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 10:
            return {}
        
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            ols_result = RegressionAnalytics.calculate_ols_hedge_ratio(
                df['x'], df['y'], add_constant=False
            )
            if not ols_result:
                return {}
            hedge_ratio = ols_result['hedge_ratio']
        
        # Calculate spread
        spread = df['y'] - hedge_ratio * df['x']
        
        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        if spread_std == 0:
            zscore = 0
        else:
            zscore = (spread - spread_mean) / spread_std
        
        current_zscore = float(zscore.iloc[-1]) if len(zscore) > 0 else 0
        
        # Trading signals
        signal = 0
        if current_zscore > self.entry_zscore:
            signal = -1  # Short spread (sell y, buy x)
        elif current_zscore < -self.entry_zscore:
            signal = 1   # Long spread (buy y, sell x)
        elif abs(current_zscore) < self.exit_zscore:
            signal = 0   # Close position
        
        return {
            'spread': spread.tolist(),
            'zscore': zscore.tolist(),
            'current_spread': float(spread.iloc[-1]) if len(spread) > 0 else 0,
            'current_zscore': current_zscore,
            'spread_mean': float(spread_mean),
            'spread_std': float(spread_std),
            'hedge_ratio': float(hedge_ratio),
            'signal': signal,
            'position': 'Long' if signal == 1 else 'Short' if signal == -1 else 'Flat',
            'entry_threshold': self.entry_zscore,
            'exit_threshold': self.exit_zscore
        }
    
    def backtest_mean_reversion(self, x: pd.Series, y: pd.Series,
                               initial_capital: float = 10000.0,
                               transaction_cost: float = 0.001) -> Dict:
        """Backtest mean reversion strategy"""
        if len(x) < 100 or len(y) < 100:
            return {}
        
        # Align series
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 100:
            return {}
        
        # Calculate rolling hedge ratio (60-day window)
        hedge_ratios = []
        for i in range(60, len(df)):
            window_x = df['x'].iloc[i-60:i]
            window_y = df['y'].iloc[i-60:i]
            result = RegressionAnalytics.calculate_ols_hedge_ratio(
                window_x, window_y, add_constant=False
            )
            if result:
                hedge_ratios.append(result['hedge_ratio'])
            else:
                hedge_ratios.append(hedge_ratios[-1] if hedge_ratios else 1)
        
        # Pad beginning
        hedge_ratios = [hedge_ratios[0]] * 60 + hedge_ratios
        
        # Calculate spread and z-score
        spread = df['y'] - np.array(hedge_ratios) * df['x']
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Initialize backtest
        capital = initial_capital
        position = 0  # 1: long spread, -1: short spread, 0: flat
        trades = []
        equity_curve = [capital]
        
        for i in range(60, len(df)):
            current_zscore = zscore.iloc[i]
            
            # Trading logic
            new_position = position
            if position == 0:
                if current_zscore > self.entry_zscore:
                    new_position = -1
                elif current_zscore < -self.entry_zscore:
                    new_position = 1
            else:
                if (position == 1 and current_zscore > -self.exit_zscore) or \
                   (position == -1 and current_zscore < self.exit_zscore):
                    new_position = 0
            
            # Execute trade if position changed
            if new_position != position:
                # Calculate P&L from previous position
                if position != 0:
                    # Simplified P&L calculation
                    price_change_x = df['x'].iloc[i] - df['x'].iloc[i-1]
                    price_change_y = df['y'].iloc[i] - df['y'].iloc[i-1]
                    
                    if position == 1:  # Long spread
                        pnl = (price_change_y - hedge_ratios[i] * price_change_x)
                    else:  # Short spread
                        pnl = -(price_change_y - hedge_ratios[i] * price_change_x)
                    
                    capital += pnl - abs(pnl) * transaction_cost
                
                position = new_position
                
                trades.append({
                    'timestamp': df.index[i],
                    'position': position,
                    'capital': capital,
                    'zscore': float(current_zscore)
                })
            
            equity_curve.append(capital)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = (equity_series.expanding().max() - equity_series).max() / equity_series.expanding().max()
            total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
        else:
            sharpe = 0
            max_drawdown = 0
            total_return = 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': float(equity_series.iloc[-1]) if len(equity_series) > 0 else initial_capital,
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'num_trades': len(trades),
            'win_rate': len([t for t in trades if t['capital'] > 0]) / len(trades) if trades else 0,
            'equity_curve': equity_curve,
            'trades': trades,
            'zscore_series': zscore.tolist()
        }
    
    def calculate_cross_correlation_matrix(self, prices: Dict[str, pd.Series],
                                          window: int = 20) -> pd.DataFrame:
        """Calculate rolling correlation matrix between multiple symbols"""
        symbols = list(prices.keys())
        n = len(symbols)
        
        if n < 2:
            return pd.DataFrame()
        
        # Align all series
        aligned = pd.DataFrame(prices).dropna()
        
        if len(aligned) < window:
            return pd.DataFrame()
        
        # Initialize correlation matrix
        corr_matrix = pd.DataFrame(np.eye(n), index=symbols, columns=symbols)
        
        # Calculate correlations
        for i in range(n):
            for j in range(i+1, n):
                corr = aligned[symbols[i]].rolling(window=window).corr(aligned[symbols[j]])
                current_corr = corr.iloc[-1] if len(corr) > 0 else 0
                corr_matrix.iloc[i, j] = current_corr
                corr_matrix.iloc[j, i] = current_corr
        
        return corr_matrix