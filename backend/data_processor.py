import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

from database import redis_client, get_async_db
from storage.tick_storage import TickStorage
from analytics.price_stats import PriceStatistics
from analytics.regression import RegressionAnalytics
from analytics.mean_reversion import MeanReversionAnalytics
from analytics.statistical_tests import StatisticalTests
from analytics.backtest import MeanReversionBacktest
from config import settings
from models.tick_data import TickData
from sqlalchemy import select, desc

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing engine"""
    
    def __init__(self):
        self.storage = TickStorage()
        self.price_stats = PriceStatistics()
        self.regression = RegressionAnalytics()
        self.mean_reversion = MeanReversionAnalytics(
            entry_zscore=settings.zscore_threshold,
            exit_zscore=0.0
        )
        self.stat_tests = StatisticalTests()
        self.analytics_cache = defaultdict(dict)
        self.running = False
        
    async def process_tick(self, tick: Dict):
        """Process incoming tick data"""
        try:
            symbol = tick['symbol']
            timestamp = tick['timestamp']
            price = tick['price']
            
            # Update Redis cache with latest price
            redis_key = f"price:{symbol}"
            redis_client.hset(redis_key, mapping={
                'price': price,
                'timestamp': timestamp.isoformat(),
                'size': tick['size']
            })
            redis_client.expire(redis_key, 3600)  # 1 hour expiry
            
            # Update symbol set
            redis_client.sadd('symbols', symbol)
            
            # Update rolling statistics
            await self._update_rolling_stats(symbol, tick)
            
            # Check alerts
            await self._check_alerts(symbol, tick)
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    async def _update_rolling_stats(self, symbol: str, tick: Dict):
        """Update rolling statistics in Redis"""
        try:
            # Get recent prices for this symbol
            recent_ticks = await self.storage.get_recent_ticks(symbol, limit=100)
            
            if not recent_ticks:
                return
            
            prices = pd.Series([t['price'] for t in recent_ticks])
            
            # Calculate rolling statistics
            for window in settings.rolling_window_sizes:
                if len(prices) >= window:
                    rolling_mean = prices.rolling(window=window).mean().iloc[-1]
                    rolling_std = prices.rolling(window=window).std().iloc[-1]
                    
                    # Store in Redis
                    key = f"stats:{symbol}:{window}"
                    redis_client.hset(key, mapping={
                        'mean': rolling_mean,
                        'std': rolling_std,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    redis_client.expire(key, 300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error updating rolling stats: {e}")
    
    async def _check_alerts(self, symbol: str, tick: Dict):
        """Check configured alerts"""
        try:
            # Get all alerts for this symbol
            alert_keys = redis_client.keys(f"alert:{symbol}:*")
            
            for alert_key in alert_keys:
                alert = redis_client.hgetall(alert_key)
                
                if not alert:
                    continue
                
                alert_type = alert.get('type')
                threshold = float(alert.get('threshold', 0))
                current_value = 0
                
                # Calculate value based on alert type
                if alert_type == 'price':
                    current_value = tick['price']
                elif alert_type == 'zscore':
                    # Need to calculate z-score
                    # This would require pair information
                    pass
                elif alert_type == 'volume':
                    current_value = tick['size']
                
                # Check condition
                condition = alert.get('condition', '>')
                triggered = False
                
                if condition == '>' and current_value > threshold:
                    triggered = True
                elif condition == '<' and current_value < threshold:
                    triggered = True
                elif condition == '==' and abs(current_value - threshold) < 1e-6:
                    triggered = True
                
                if triggered:
                    # Store alert event
                    import json
                    alert_event = {
                        'symbol': symbol,
                        'alert_type': alert_type,
                        'threshold': threshold,
                        'current_value': current_value,
                        'timestamp': datetime.utcnow().isoformat(),
                        'message': alert.get('message', 'Alert triggered')
                    }
                    
                    redis_client.lpush('alerts:triggered', json.dumps(alert_event))
                    redis_client.ltrim('alerts:triggered', 0, 99)  # Keep last 100
                    
                    # Send notification (could be extended to email/webhook)
                    logger.info(f"Alert triggered: {alert_event}")
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def calculate_pair_analytics(self, symbol1: str, symbol2: str, 
                                      timeframe: str = '1m') -> Dict:
        """Calculate analytics for a pair of symbols"""
        try:
            # Get resampled data
            data1 = await self.storage.get_resampled_data(symbol1, timeframe)
            data2 = await self.storage.get_resampled_data(symbol2, timeframe)
            
            if data1.empty or data2.empty:
                return {}
            
            # Align timestamps
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) < 10:
                return {}
            
            data1_aligned = data1.loc[common_idx]
            data2_aligned = data2.loc[common_idx]
            
            # Calculate analytics
            results = {}
            
            # OLS regression
            ols_result = self.regression.calculate_ols_hedge_ratio(
                data1_aligned['close'], data2_aligned['close']
            )
            if ols_result:
                results['ols'] = ols_result
            
            # Robust regression
            huber_result = self.regression.calculate_robust_regression(
                data1_aligned['close'], data2_aligned['close'], method='huber'
            )
            if huber_result:
                results['huber'] = huber_result
            
            # Kalman filter
            kalman_result = self.regression.calculate_kalman_filter_hedge(
                data1_aligned['close'], data2_aligned['close']
            )
            if kalman_result:
                results['kalman'] = kalman_result
            
            # Mean reversion analytics
            if 'ols' in results:
                spread_result = self.mean_reversion.calculate_spread_zscore(
                    data1_aligned['close'], data2_aligned['close'],
                    results['ols']['hedge_ratio']
                )
                if spread_result:
                    results['spread'] = spread_result
            
            # Statistical tests
            adf_result = self.stat_tests.adf_test(
                pd.Series(results.get('spread', {}).get('spread', []))
            )
            if adf_result:
                results['adf_test'] = adf_result
            
            coint_result = self.stat_tests.cointegration_test(
                data1_aligned['close'], data2_aligned['close']
            )
            if coint_result:
                results['cointegration'] = coint_result
            
            # Rolling correlation
            corr_series = self.regression.calculate_rolling_correlation(
                data1_aligned['close'], data2_aligned['close'], window=20
            )
            if not corr_series.empty:
                results['rolling_correlation'] = {
                    'values': corr_series.tolist(),
                    'timestamps': corr_series.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'current': float(corr_series.iloc[-1]) if len(corr_series) > 0 else 0
                }
            
            # Store in cache
            cache_key = f"{symbol1}:{symbol2}:{timeframe}"
            self.analytics_cache[cache_key] = {
                'results': results,
                'timestamp': datetime.utcnow()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating pair analytics: {e}")
            return {}
    
    async def get_symbol_analytics(self, symbol: str, timeframe: str = '1m') -> Dict:
        """Get analytics for a single symbol"""
        try:
            data = await self.storage.get_resampled_data(symbol, timeframe)
            
            if data.empty:
                return {}
            
            results = {}
            
            # Price statistics
            if not data['close'].empty:
                basic_stats = self.price_stats.calculate_basic_stats(data['close'])
                if basic_stats:
                    results['price_stats'] = basic_stats
                
                rolling_stats = self.price_stats.calculate_rolling_stats(data['close'])
                if not rolling_stats.empty:
                    results['rolling_stats'] = rolling_stats.to_dict()
            
            # Volume analysis
            if not data['volume'].empty:
                volume_stats = {
                    'current_volume': float(data['volume'].iloc[-1]) if len(data['volume']) > 0 else 0,
                    'volume_mean': float(data['volume'].mean()),
                    'volume_std': float(data['volume'].std()),
                    'volume_sum': float(data['volume'].sum())
                }
                results['volume_stats'] = volume_stats
            
            # Market quality
            if not data['close'].empty and not data['volume'].empty:
                market_quality = self.price_stats.calculate_market_quality(
                    data['close'], data['volume']
                )
                if market_quality:
                    results['market_quality'] = market_quality
            
            # Statistical tests
            if not data['close'].empty:
                hurst = self.stat_tests.calculate_hurst_exponent(data['close'])
                results['hurst_exponent'] = hurst
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating symbol analytics: {e}")
            return {}
    
    async def export_data(self, symbol: str, timeframe: str = None,
                         start_time: datetime = None, 
                         end_time: datetime = None) -> pd.DataFrame:
        """Export data for download"""
        try:
            if timeframe:
                # Export resampled data
                data = await self.storage.get_resampled_data(
                    symbol, timeframe, start_time, end_time
                )
            else:
                # Export tick data (limit to 10000 rows)
                async with get_async_db() as session:
                    stmt = (
                        select(TickData)
                        .where(TickData.symbol == symbol)
                    )
                    
                    if start_time:
                        stmt = stmt.where(TickData.timestamp >= start_time)
                    if end_time:
                        stmt = stmt.where(TickData.timestamp <= end_time)
                    
                    stmt = stmt.order_by(TickData.timestamp).limit(10000)
                    
                    result = await session.execute(stmt)
                    ticks = result.scalars().all()
                
                if not ticks:
                    return pd.DataFrame()
                
                data = pd.DataFrame([
                    {
                        'timestamp': t.timestamp,
                        'price': t.price,
                        'size': t.size
                    }
                    for t in ticks
                ])
                
                if not data.empty:
                    data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return pd.DataFrame()