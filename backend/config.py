import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = "postgresql://quantuser:quantpass@postgres:5432/quantdb"
    redis_url: str = "redis://redis:6379/0"
    
    # WebSocket
    websocket_url: str = "wss://fstream.binance.com/ws/{symbol}@trade"
    symbols: List[str] = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]
    
    # Sampling intervals in seconds
    sampling_intervals: List[int] = [1, 60, 300]  # 1s, 1m, 5m
    
    # Analytics
    rolling_window_sizes: List[int] = [20, 50, 100]
    zscore_threshold: float = 2.0
    correlation_window: int = 100
    
    # Alert
    alert_check_interval: int = 1  # seconds
    
    # Storage
    max_tick_buffer: int = 10000
    flush_interval: int = 5  # seconds
    
    @validator("database_url")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://"):
            raise ValueError("Invalid database URL")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()