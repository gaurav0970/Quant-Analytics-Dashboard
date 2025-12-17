import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from collections import defaultdict
import logging

from database import get_async_db, redis_client
from models.tick_data import TickData, ResampledData
from config import settings

logger = logging.getLogger(__name__)

class TickStorage:
    """Storage and processing for tick data"""
    
    def __init__(self):
        self.tick_buffer = defaultdict(list)
        self.max_buffer = settings.max_tick_buffer
        self.flush_interval = settings.flush_interval
        
    async def store_tick(self, tick: Dict):
        """Store tick in buffer and periodically flush to database"""
        symbol = tick['symbol']
        
        # Add to buffer
        self.tick_buffer[symbol].append(tick)
        
        # Check if buffer needs flushing
        if len(self.tick_buffer[symbol]) >= self.max_buffer:
            await self._flush_buffer(symbol)
    
    async def _flush_buffer(self, symbol: str):
        """Flush buffer to database"""
        if symbol not in self.tick_buffer or not self.tick_buffer[symbol]:
            return
        
        ticks = self.tick_buffer[symbol]
        async with get_async_db() as session:
            for tick in ticks:
                db_tick = TickData(
                    symbol=tick['symbol'],
                    timestamp=tick['timestamp'],
                    price=tick['price'],
                    size=tick['size']
                )
                session.add(db_tick)
            
            await session.commit()
        
        # Clear buffer
        self.tick_buffer[symbol] = []
        logger.info(f"Flushed {len(ticks)} ticks for {symbol} to database")
    
    async def get_recent_ticks(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Get recent ticks for a symbol"""
        async with get_async_db() as session:
            stmt = (
                select(TickData)
                .where(TickData.symbol == symbol)
                .order_by(desc(TickData.timestamp))
                .limit(limit)
            )
            result = await session.execute(stmt)
            ticks = result.scalars().all()
            
            return [
                {
                    'timestamp': tick.timestamp,
                    'price': tick.price,
                    'size': tick.size
                }
                for tick in ticks
            ]
    
    async def resample_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Resample tick data to specified timeframe"""
        async with get_async_db() as session:
            # Get data for last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            stmt = (
                select(TickData)
                .where(TickData.symbol == symbol)
                .where(TickData.timestamp >= one_hour_ago)
                .order_by(TickData.timestamp)
            )
            result = await session.execute(stmt)
            ticks = result.scalars().all()
        
        if not ticks:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'price': t.price,
                'size': t.size
            }
            for t in ticks
        ])
        
        df.set_index('timestamp', inplace=True)
        
        # Resample
        if timeframe == '1s':
            resampled = df['price'].resample('1S').ohlc()
            volume = df['size'].resample('1S').sum()
        elif timeframe == '1m':
            resampled = df['price'].resample('1T').ohlc()
            volume = df['size'].resample('1T').sum()
        elif timeframe == '5m':
            resampled = df['price'].resample('5T').ohlc()
            volume = df['size'].resample('5T').sum()
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Calculate VWAP
        df['price_size'] = df['price'] * df['size']
        if timeframe == '1s':
            price_size_sum = df['price_size'].resample('1S').sum()
            size_sum = df['size'].resample('1S').sum()
        elif timeframe == '1m':
            price_size_sum = df['price_size'].resample('1T').sum()
            size_sum = df['size'].resample('1T').sum()
        else:  # 5m
            price_size_sum = df['price_size'].resample('5T').sum()
            size_sum = df['size'].resample('5T').sum()
        
        vwap = price_size_sum / size_sum
        
        # Combine results
        result_df = pd.DataFrame({
            'open': resampled['open'],
            'high': resampled['high'],
            'low': resampled['low'],
            'close': resampled['close'],
            'volume': volume,
            'vwap': vwap
        }).dropna()
        
        # Store resampled data
        await self._store_resampled_data(symbol, timeframe, result_df)
        
        return result_df
    
    async def _store_resampled_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store resampled data in database"""
        async with get_async_db() as session:
            for idx, row in df.iterrows():
                resampled = ResampledData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=idx.to_pydatetime(),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    vwap=row.get('vwap', 0)
                )
                session.add(resampled)
            
            await session.commit()
    
    async def get_resampled_data(self, symbol: str, timeframe: str, 
                                start_time: datetime = None, 
                                end_time: datetime = None) -> pd.DataFrame:
        """Get resampled data from database"""
        async with get_async_db() as session:
            stmt = (
                select(ResampledData)
                .where(ResampledData.symbol == symbol)
                .where(ResampledData.timeframe == timeframe)
            )
            
            if start_time:
                stmt = stmt.where(ResampledData.timestamp >= start_time)
            if end_time:
                stmt = stmt.where(ResampledData.timestamp <= end_time)
            
            stmt = stmt.order_by(ResampledData.timestamp)
            
            result = await session.execute(stmt)
            data = result.scalars().all()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume,
                'vwap': d.vwap
            }
            for d in data
        ])
        
        df.set_index('timestamp', inplace=True)
        return df