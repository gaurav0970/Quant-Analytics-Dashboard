import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MeanReversionBacktest:
    """Backtest mean reversion strategies on spread data"""
    
    def __init__(self, entry_zscore: float = 2.0, exit_zscore: float = 0.5,
                 stop_loss_zscore: float = 4.0):
        """
        Initialize backtest parameters
        
        Args:
            entry_zscore: Z-score threshold for entering positions
            exit_zscore: Z-score threshold for exiting positions
            stop_loss_zscore: Maximum z-score before stop loss
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
    
    def run_backtest(self, spread: pd.Series, z_score: pd.Series,
                    price1: pd.Series, price2: pd.Series,
                    hedge_ratio: float = 1.0) -> Dict:
        """
        Run mean reversion backtest
        
        Strategy logic:
        - Enter long spread when z-score < -entry_zscore
        - Enter short spread when z-score > entry_zscore
        - Exit when abs(z-score) < exit_zscore
        - Stop loss when abs(z-score) > stop_loss_zscore
        
        Args:
            spread: Spread series
            z_score: Z-score series
            price1: Price series for asset 1
            price2: Price series for asset 2
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if len(spread) < 10 or len(z_score) < 10:
                return {}
            
            # Align all series
            df = pd.DataFrame({
                'spread': spread,
                'zscore': z_score,
                'price1': price1,
                'price2': price2
            }).dropna()
            
            if len(df) < 10:
                return {}
            
            # Initialize tracking variables
            position = 0  # 1 = long spread, -1 = short spread, 0 = flat
            entry_price = 0
            entry_zscore = 0
            trades = []
            equity_curve = [1.0]  # Start with $1
            current_equity = 1.0
            
            returns = []
            
            for i in range(len(df)):
                zscore_val = df['zscore'].iloc[i]
                spread_val = df['spread'].iloc[i]
                
                # Check for entry signal
                if position == 0:
                    # Long spread entry (buy asset1, sell asset2)
                    if zscore_val < -self.entry_zscore:
                        position = 1
                        entry_price = spread_val
                        entry_zscore = zscore_val
                        entry_idx = i
                    
                    # Short spread entry (sell asset1, buy asset2)
                    elif zscore_val > self.entry_zscore:
                        position = -1
                        entry_price = spread_val
                        entry_zscore = zscore_val
                        entry_idx = i
                
                # Check for exit signal
                elif position != 0:
                    exit_signal = False
                    exit_reason = None
                    
                    # Normal exit: mean reversion
                    if abs(zscore_val) < self.exit_zscore:
                        exit_signal = True
                        exit_reason = 'mean_revert'
                    
                    # Stop loss exit
                    elif abs(zscore_val) > self.stop_loss_zscore:
                        exit_signal = True
                        exit_reason = 'stop_loss'
                    
                    # Exit on opposite extreme signal
                    elif position == 1 and zscore_val > self.entry_zscore:
                        exit_signal = True
                        exit_reason = 'reversal'
                    elif position == -1 and zscore_val < -self.entry_zscore:
                        exit_signal = True
                        exit_reason = 'reversal'
                    
                    if exit_signal:
                        # Calculate P&L
                        pnl = position * (spread_val - entry_price)
                        ret = pnl / abs(entry_price) if abs(entry_price) > 0 else 0
                        
                        # Update equity
                        current_equity *= (1 + ret)
                        returns.append(ret)
                        
                        # Record trade
                        trades.append({
                            'entry_index': entry_idx,
                            'exit_index': i,
                            'entry_timestamp': df.index[entry_idx],
                            'exit_timestamp': df.index[i],
                            'position': 'long_spread' if position == 1 else 'short_spread',
                            'entry_spread': entry_price,
                            'exit_spread': spread_val,
                            'entry_zscore': entry_zscore,
                            'exit_zscore': zscore_val,
                            'pnl': pnl,
                            'return': ret * 100,
                            'holding_periods': i - entry_idx,
                            'exit_reason': exit_reason
                        })
                        
                        # Reset position
                        position = 0
                        entry_price = 0
                
                equity_curve.append(current_equity)
            
            # Calculate performance metrics
            if len(returns) == 0:
                return {
                    'total_trades': 0,
                    'message': 'No trades executed - no entry signals met threshold'
                }
            
            returns_arr = np.array(returns)
            
            # Basic metrics
            total_return = (current_equity - 1.0) * 100
            win_rate = np.sum(returns_arr > 0) / len(returns_arr) * 100 if len(returns_arr) > 0 else 0
            avg_return = np.mean(returns_arr) * 100
            max_return = np.max(returns_arr) * 100
            min_return = np.min(returns_arr) * 100
            
            # Risk metrics
            returns_std = np.std(returns_arr)
            sharpe_ratio = np.mean(returns_arr) / returns_std * np.sqrt(252) if returns_std > 0 else 0
            
            # Drawdown calculation
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            # Winning/Losing streaks
            winning_trades = [t for t in trades if t['return'] > 0]
            losing_trades = [t for t in trades if t['return'] < 0]
            
            avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Average holding period
            avg_holding = np.mean([t['holding_periods'] for t in trades])
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': float(win_rate),
                'total_return': float(total_return),
                'avg_return_per_trade': float(avg_return),
                'max_return': float(max_return),
                'min_return': float(min_return),
                'avg_winning_trade': float(avg_win),
                'avg_losing_trade': float(avg_loss),
                'profit_factor': float(profit_factor),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'avg_holding_periods': float(avg_holding),
                'final_equity': float(current_equity),
                'equity_curve': [float(e) for e in equity_curve],
                'trades': trades,
                'parameters': {
                    'entry_zscore': self.entry_zscore,
                    'exit_zscore': self.exit_zscore,
                    'stop_loss_zscore': self.stop_loss_zscore
                }
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def optimize_parameters(self, spread: pd.Series, z_score: pd.Series,
                          price1: pd.Series, price2: pd.Series,
                          hedge_ratio: float = 1.0,
                          entry_range: Tuple[float, float] = (1.5, 3.0),
                          exit_range: Tuple[float, float] = (0.25, 1.0)) -> Dict:
        """
        Optimize entry and exit z-score thresholds
        
        Args:
            spread, z_score, price1, price2: Price data
            hedge_ratio: Hedge ratio for the pair
            entry_range: Range of entry z-scores to test
            exit_range: Range of exit z-scores to test
            
        Returns:
            Best parameters and results
        """
        try:
            best_sharpe = -np.inf
            best_params = None
            best_result = None
            
            # Grid search
            entry_steps = np.linspace(entry_range[0], entry_range[1], 5)
            exit_steps = np.linspace(exit_range[0], exit_range[1], 5)
            
            all_results = []
            
            for entry_z in entry_steps:
                for exit_z in exit_steps:
                    # Run backtest with these parameters
                    backtester = MeanReversionBacktest(
                        entry_zscore=entry_z,
                        exit_zscore=exit_z
                    )
                    
                    result = backtester.run_backtest(
                        spread, z_score, price1, price2, hedge_ratio
                    )
                    
                    if result and 'sharpe_ratio' in result:
                        sharpe = result['sharpe_ratio']
                        
                        all_results.append({
                            'entry_zscore': entry_z,
                            'exit_zscore': exit_z,
                            'sharpe_ratio': sharpe,
                            'total_return': result['total_return'],
                            'win_rate': result['win_rate'],
                            'total_trades': result['total_trades']
                        })
                        
                        if sharpe > best_sharpe and result['total_trades'] >= 5:
                            best_sharpe = sharpe
                            best_params = {'entry': entry_z, 'exit': exit_z}
                            best_result = result
            
            return {
                'best_parameters': best_params,
                'best_result': best_result,
                'all_results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
