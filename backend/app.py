import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from contextlib import asynccontextmanager
import os
from typing import List, Optional  # ADD THIS LINE

from config import settings
from database import init_database, test_connections, get_async_db
from websocket_client import BinanceWebSocketClient
from data_processor import DataProcessor
from models.tick_data import TickData, ResampledData
from sqlalchemy import func, desc, select  # ADD select

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
websocket_client = None
data_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    # Startup
    logger.info("Starting up...")
    
    # Initialize database
    await init_database()
    
    # Test connections
    await test_connections()
    
    # Initialize WebSocket client
    global websocket_client, data_processor
    data_processor = DataProcessor()
    websocket_client = BinanceWebSocketClient(settings.symbols)
    websocket_client.register_callback(data_processor.process_tick)
    
    # Start WebSocket client in background
    asyncio.create_task(websocket_client.start())
    
    # Start periodic tasks
    asyncio.create_task(periodic_resampling())
    asyncio.create_task(periodic_analytics())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if websocket_client:
        await websocket_client.stop()

# Create FastAPI app
app = FastAPI(
    title="Quant Analytics Platform",
    description="Real-time crypto analytics platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
active_connections: List[WebSocket] = []  # Add type annotation

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

async def periodic_resampling():
    """Periodically resample data for all timeframes"""
    while True:
        try:
            for symbol in settings.symbols:
                for timeframe in ['1s', '1m', '5m']:
                    await data_processor.storage.resample_data(symbol, timeframe)
            
            await asyncio.sleep(60)  # Resample every minute
        except Exception as e:
            logger.error(f"Error in periodic resampling: {e}")
            await asyncio.sleep(10)

async def periodic_analytics():
    """Periodically calculate analytics"""
    while True:
        try:
            # Calculate pair analytics for major pairs
            major_pairs = [('BTCUSDT', 'ETHUSDT'), ('BTCUSDT', 'BNBUSDT'), ('ETHUSDT', 'BNBUSDT')]
            
            for pair in major_pairs:
                await data_processor.calculate_pair_analytics(pair[0], pair[1], '1m')
            
            await asyncio.sleep(30)  # Calculate every 30 seconds
        except Exception as e:
            logger.error(f"Error in periodic analytics: {e}")
            await asyncio.sleep(10)

# API Routes
@app.get("/")
async def root():
    return {"message": "Quant Analytics Platform API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    connections_ok = websocket_client is not None and websocket_client.running
    return {
        "status": "healthy" if connections_ok else "degraded",
        "websocket": "connected" if connections_ok else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/symbols")
async def get_symbols():
    """Get available symbols"""
    return {"symbols": settings.symbols}

@app.get("/api/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a symbol"""
    from database import redis_client
    
    price_data = redis_client.hgetall(f"price:{symbol.upper()}")
    if not price_data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    return {
        "symbol": symbol.upper(),
        "price": float(price_data.get('price', 0)),
        "timestamp": price_data.get('timestamp'),
        "size": float(price_data.get('size', 0))
    }

@app.get("/api/ohlc/{symbol}")
async def get_ohlc_data(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100
):
    """Get OHLC data for a symbol"""
    data = await data_processor.storage.get_resampled_data(
        symbol.upper(), timeframe
    )
    
    if data.empty:
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "data": []
        }
    
    # Limit results
    data = data.tail(limit)
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "data": [
            {
                "timestamp": idx.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),
                "vwap": float(row.get('vwap', 0))
            }
            for idx, row in data.iterrows()
        ]
    }

@app.get("/api/analytics/symbol/{symbol}")
async def get_symbol_analytics(
    symbol: str,
    timeframe: str = "1m"
):
    """Get analytics for a single symbol"""
    results = await data_processor.get_symbol_analytics(
        symbol.upper(), timeframe
    )
    
    if not results:
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "analytics": {}
        }
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "analytics": results
    }

@app.get("/api/analytics/pair/{symbol1}/{symbol2}")
async def get_pair_analytics(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m"
):
    """Get analytics for a pair of symbols"""
    results = await data_processor.calculate_pair_analytics(
        symbol1.upper(), symbol2.upper(), timeframe
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="No analytics available")
    
    return {
        "pair": f"{symbol1.upper()}/{symbol2.upper()}",
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "analytics": results
    }

@app.post("/api/alerts")
async def create_alert(alert: dict):
    """Create a new alert"""
    from database import redis_client
    
    required_fields = ['symbol', 'type', 'condition', 'threshold']
    for field in required_fields:
        if field not in alert:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")
    
    alert_id = f"alert:{alert['symbol']}:{datetime.utcnow().timestamp()}"
    
    redis_client.hset(alert_id, mapping=alert)
    redis_client.expire(alert_id, 86400)  # 24 hours
    
    return {"id": alert_id, "message": "Alert created successfully"}

@app.get("/api/alerts/triggered")
async def get_triggered_alerts(limit: int = 10):
    """Get triggered alerts"""
    from database import redis_client
    
    try:
        alerts = redis_client.lrange('alerts:triggered', 0, limit - 1)
        parsed_alerts = []
        for alert in alerts:
            try:
                if isinstance(alert, bytes):
                    alert = alert.decode('utf-8')
                parsed_alerts.append(json.loads(alert))
            except (json.JSONDecodeError, ValueError):
                continue
        
        return {
            "alerts": parsed_alerts
        }
    except Exception as e:
        logger.error(f"Error fetching triggered alerts: {e}")
        return {"alerts": []}


@app.get("/api/export/{symbol}")
async def export_data(
    symbol: str,
    timeframe: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Export data as CSV"""
    try:
        # Parse datetime parameters
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Get data
        data = await data_processor.export_data(
            symbol.upper(), timeframe, start_dt, end_dt
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Convert to CSV
        csv_data = data.to_csv()
        
        # Create response
        filename = f"export_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return JSONResponse(
            content={"data": csv_data, "filename": filename},
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(params: dict):
    """Run mean reversion backtest"""
    try:
        from analytics.backtest import MeanReversionBacktest
        
        # Required parameters
        symbol1 = params.get('symbol1')
        symbol2 = params.get('symbol2')
        timeframe = params.get('timeframe', '1m')
        
        if not symbol1 or not symbol2:
            raise HTTPException(status_code=400, detail="Missing symbol1 or symbol2")
        
        # Get pair analytics first
        analytics = await data_processor.calculate_pair_analytics(
            symbol1.upper(), symbol2.upper(), timeframe
        )
        
        if not analytics or 'spread' not in analytics:
            raise HTTPException(status_code=404, detail="Insufficient data for backtest")
        
        # Get price data
        data1 = await data_processor.storage.get_resampled_data(symbol1.upper(), timeframe)
        data2 = await data_processor.storage.get_resampled_data(symbol2.upper(), timeframe)
        
        if data1.empty or data2.empty:
            raise HTTPException(status_code=404, detail="Price data not available")
        
        # Align data
        common_idx = data1.index.intersection(data2.index)
        data1_aligned = data1.loc[common_idx]
        data2_aligned = data2.loc[common_idx]
        
        # Extract parameters
        entry_zscore = params.get('entry_zscore', 2.0)
        exit_zscore = params.get('exit_zscore', 0.5)
        stop_loss = params.get('stop_loss_zscore', 4.0)
        
        # Use OLS hedge ratio
        hedge_ratio = analytics.get('ols', {}).get('hedge_ratio', 1.0)
        
        # Calculate spread and z-score
        spread_data = analytics['spread']
        spread_series = pd.Series(spread_data['spread'], index=pd.to_datetime(spread_data['timestamps']))
        zscore_series = pd.Series(spread_data['zscore'], index=pd.to_datetime(spread_data['timestamps']))
        
        # Run backtest
        backtester = MeanReversionBacktest(
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_loss_zscore=stop_loss
        )
        
        results = backtester.run_backtest(
            spread_series,
            zscore_series,
            data1_aligned['close'],
            data2_aligned['close'],
            hedge_ratio
        )
        
        return {
            "pair": f"{symbol1}/{symbol2}",
            "timeframe": timeframe,
            "hedge_ratio": hedge_ratio,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/ohlc")
async def upload_ohlc(data: dict):
    """Upload OHLC data for analysis"""
    try:
        from models.tick_data import ResampledData
        
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1m')
        ohlc_data = data.get('data')  # List of {timestamp, open, high, low, close, volume}
        
        if not symbol or not ohlc_data:
            raise HTTPException(status_code=400, detail="Missing symbol or data")
        
        # Store in database
        async with get_async_db() as session:
            records = []
            for row in ohlc_data:
                record = ResampledData(
                    symbol=symbol.upper(),
                    timeframe=timeframe,
                    timestamp=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row.get('volume', 0))
                )
                records.append(record)
            
            session.add_all(records)
            await session.commit()
        
        return {
            "message": "OHLC data uploaded successfully",
            "symbol": symbol,
            "records": len(records)
        }
        
    except Exception as e:
        logger.error(f"Error uploading OHLC data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    from database import redis_client, get_async_db
    from sqlalchemy import func, select
    from models.tick_data import TickData
    
    stats = {}
    
    try:
        # Get database stats using async session
        async with get_async_db() as session:
            # Tick count
            stmt = select(func.count(TickData.id))
            result = await session.execute(stmt)
            tick_count = result.scalar()
            stats['tick_count'] = tick_count
            
            # Symbol counts
            stmt = (
                select(TickData.symbol, func.count(TickData.id))
                .group_by(TickData.symbol)
            )
            result = await session.execute(stmt)
            symbol_counts = result.all()
            stats['symbol_counts'] = dict(symbol_counts)
            
            # Latest ticks
            stmt = (
                select(TickData)
                .order_by(desc(TickData.timestamp))
                .limit(5)
            )
            result = await session.execute(stmt)
            latest_ticks = result.scalars().all()
            stats['latest_ticks'] = [
                {
                    'symbol': t.symbol,
                    'price': t.price,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in latest_ticks
            ]
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        stats['database_error'] = str(e)
        stats['tick_count'] = 0
        stats['symbol_counts'] = {}
        stats['latest_ticks'] = []
    
    # Redis stats
    try:
        stats['redis_connected'] = redis_client.ping()
        stats['active_symbols'] = list(redis_client.smembers('symbols'))
    except Exception as e:
        logger.error(f"Error getting Redis stats: {e}")
        stats['redis_error'] = str(e)
    
    # WebSocket stats
    stats['websocket_running'] = websocket_client.running if websocket_client else False
    stats['websocket_symbols'] = websocket_client.symbols if websocket_client else []
    
    return stats

# WebSocket endpoint for real-time updates
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Broadcast updates
async def broadcast_update(update_type: str, data: dict):
    """Broadcast update to all WebSocket clients"""
    message = {
        "type": update_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(message)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting backend on {host}:{port}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )