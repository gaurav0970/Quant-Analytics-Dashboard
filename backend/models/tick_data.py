from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pandas as pd

Base = declarative_base()

class TickData(Base):
    """Raw tick data model"""
    __tablename__ = "tick_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'size': self.size
        }

class ResampledData(Base):
    """Resampled OHLC data"""
    __tablename__ = "resampled_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1s', '1m', '5m'
    timestamp = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float)  # Volume Weighted Average Price
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )

class AnalyticsResult(Base):
    """Analytics results storage"""
    __tablename__ = "analytics_results"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol1 = Column(String(20), index=True)
    symbol2 = Column(String(20), index=True)
    timestamp = Column(DateTime, index=True)
    analytics_type = Column(String(50))  # 'zscore', 'correlation', 'hedge_ratio', etc.
    value = Column(Float)
    parameters = Column(String)  # JSON string of parameters
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)