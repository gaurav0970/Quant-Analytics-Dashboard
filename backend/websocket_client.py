import asyncio
import json
import websockets
from datetime import datetime
import aiohttp
from typing import Dict, List, Optional, Callable
import logging
from contextlib import asynccontextmanager

from config import settings
from models.tick_data import TickData
from storage.tick_storage import TickStorage

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """Binance WebSocket client for real-time trade data"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or settings.symbols
        self.websocket_url = settings.websocket_url
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.running = False
        self.callbacks: List[Callable] = []
        self.storage = TickStorage()
        
    def register_callback(self, callback: Callable):
        """Register callback for tick data"""
        self.callbacks.append(callback)
    
    async def _handle_trade(self, symbol: str, data: dict):
        """Process trade data"""
        try:
            tick = {
                'symbol': symbol.upper(),
                'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                'price': float(data['p']),
                'size': float(data['q']),
                'trade_id': data['t'],
                'is_buyer_maker': data['m']
            }
            
            # Store tick
            await self.storage.store_tick(tick)
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    await callback(tick)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")
    
    async def _connect_symbol(self, symbol: str):
        """Connect to WebSocket for a single symbol"""
        url = self.websocket_url.format(symbol=symbol)
        
        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info(f"Connected to {symbol} WebSocket")
                    self.connections[symbol] = websocket
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            if data.get('e') == 'trade':
                                await self._handle_trade(symbol, data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection closed for {symbol}, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start WebSocket connections for all symbols"""
        self.running = True
        logger.info(f"Starting WebSocket connections for symbols: {self.symbols}")
        
        # Create tasks for each symbol
        tasks = [self._connect_symbol(symbol) for symbol in self.symbols]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("WebSocket client stopped")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        for symbol, ws in self.connections.items():
            try:
                await ws.close()
            except:
                pass
        self.connections.clear()
        logger.info("All WebSocket connections closed")
    
    async def get_historical_data(self, symbol: str, interval: str = "1m", limit: int = 1000):
        """Fetch historical data from Binance API"""
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        for candle in data
                    ]
                else:
                    raise Exception(f"API request failed: {response.status}")