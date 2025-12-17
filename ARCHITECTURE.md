# Architecture Diagram

## System Architecture - Quant Analytics Platform

### Components Overview

This document describes the architecture of the Real-Time Cryptocurrency Analytics Platform. Import the .drawio file into draw.io or use this description to recreate the diagram.

## Diagram Structure

### Layer 1: Data Source
```
┌──────────────────────────────────┐
│   Binance WebSocket API          │
│   (Real-time Tick Data Stream)   │
│                                   │
│   Endpoints:                      │
│   - wss://fstream.binance.com/   │
│   - Trade events: {               │
│      symbol, timestamp,           │
│      price, size                  │
│   }                               │
└──────────────┬───────────────────┘
               │
               │ WebSocket Connection
               │ (Persistent, Auto-reconnect)
               ▼
```

### Layer 2: Backend Application
```
┌────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                         │
│   Host: 0.0.0.0:8000                                        │
│   Runtime: Python 3.11 + Uvicorn ASGI Server                │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  WebSocket Client Module                             │  │
│  │  (websocket_client.py)                               │  │
│  │                                                       │  │
│  │  - Manages multiple symbol streams                   │  │
│  │  - Handles reconnection logic                        │  │
│  │  - Parses incoming tick data                         │  │
│  │  - Registers callbacks for processing                │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│                   ▼                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data Processor Engine                               │  │
│  │  (data_processor.py)                                 │  │
│  │                                                       │  │
│  │  Core Functions:                                     │  │
│  │  • process_tick(tick_data)                           │  │
│  │  • update_rolling_stats()                            │  │
│  │  • check_alerts()                                    │  │
│  │  • calculate_pair_analytics()                        │  │
│  │  • run_backtest()                                    │  │
│  └──────────┬──────────────────────────┬────────────────┘  │
│             │                           │                    │
│             ▼                           ▼                    │
│  ┌─────────────────────┐   ┌──────────────────────────┐    │
│  │  Tick Storage       │   │  Analytics Modules       │    │
│  │  (tick_storage.py)  │   │                          │    │
│  │                     │   │  • price_stats.py        │    │
│  │  • Store ticks      │   │  • regression.py         │    │
│  │  • Resample OHLCV   │   │    - OLS                 │    │
│  │  • Query by symbol  │   │    - Kalman Filter       │    │
│  │  • Time-series ops  │   │    - Robust (Huber)      │    │
│  └──────────┬──────────┘   │  • mean_reversion.py     │    │
│             │               │  • statistical_tests.py  │    │
│             │               │    - ADF Test            │    │
│             │               │    - Cointegration       │    │
│             │               │  • backtest.py           │    │
│             │               │    - Entry/Exit Logic    │    │
│             │               │    - P&L Calculation     │    │
│             │               └──────────────────────────┘    │
│             │                                                │
│  ┌──────────▼────────────────────────────────────────────┐ │
│  │  REST API Endpoints (FastAPI)                         │ │
│  │                                                        │ │
│  │  GET  /api/price/{symbol}         - Current price    │ │
│  │  GET  /api/ohlc/{symbol}          - OHLCV data       │ │
│  │  GET  /api/analytics/symbol       - Single analytics │ │
│  │  GET  /api/analytics/pair         - Pair analytics   │ │
│  │  POST /api/backtest               - Run backtest     │ │
│  │  POST /api/alerts                 - Create alert     │ │
│  │  GET  /api/alerts/triggered       - Get alerts       │ │
│  │  GET  /api/export/{symbol}        - Export CSV       │ │
│  │  POST /api/upload/ohlc            - Upload data      │ │
│  │  GET  /api/stats                  - System stats     │ │
│  │  WS   /ws/updates                 - Real-time feed   │ │
│  └────────────────────────────────────────────────────── │
└──────────┬──────────────────────────┬─────────────────────┘
           │                           │
           ▼                           ▼
```

### Layer 3: Data Storage
```
┌──────────────────────────┐    ┌───────────────────────────┐
│   PostgreSQL Database    │    │      Redis Cache          │
│   (Primary Storage)      │    │   (Real-time Cache)       │
│                          │    │                           │
│   Port: 5432             │    │   Port: 6379              │
│   Image: postgres:15     │    │   Image: redis:7-alpine   │
├──────────────────────────┤    ├───────────────────────────┤
│                          │    │                           │
│  Tables:                 │    │  Keys:                    │
│  ┌───────────────────┐  │    │  • price:{symbol}         │
│  │ tick_data         │  │    │  • stats:{symbol}:{win}   │
│  │─────────────────  │  │    │  • alert:{symbol}:{id}    │
│  │ id (PK)           │  │    │  • alerts:triggered       │
│  │ symbol            │  │    │  • symbols (set)          │
│  │ timestamp (idx)   │  │    │                           │
│  │ price             │  │    │  TTL:                     │
│  │ size              │  │    │  • Prices: 1 hour         │
│  └───────────────────┘  │    │  • Stats: 5 minutes       │
│                          │    │  • Alerts: 24 hours       │
│  ┌───────────────────┐  │    │                           │
│  │ resampled_data    │  │    │  Data Structures:         │
│  │─────────────────  │  │    │  • Hashes (price data)    │
│  │ id (PK)           │  │    │  • Lists (triggered)      │
│  │ symbol            │  │    │  • Sets (symbols)         │
│  │ timeframe         │  │    └───────────────────────────┘
│  │ timestamp (idx)   │  │
│  │ open, high,       │  │
│  │ low, close        │  │
│  │ volume            │  │
│  └───────────────────┘  │
│                          │
│  Indexes:                │
│  • (symbol, timestamp)   │
│  • timeframe             │
│                          │
│  Partitioning:           │
│  • By day (future)       │
└──────────────────────────┘
```

### Layer 4: Frontend Application
```
┌────────────────────────────────────────────────────────────┐
│                  FRONTEND (Streamlit)                       │
│   Host: 0.0.0.0:8501                                        │
│   Runtime: Python 3.11 + Streamlit Server                   │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Dashboard Layout (app.py)                         │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │  Tab 1: Price Monitor                        │ │    │
│  │  │  • Real-time price charts (Plotly)           │ │    │
│  │  │  • Multi-symbol candlestick charts           │ │    │
│  │  │  • Volume analysis                            │ │    │
│  │  │  • Technical indicators                       │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │  Tab 2: Pair Analytics                       │ │    │
│  │  │  • Symbol pair selection                     │ │    │
│  │  │  • Hedge ratio comparison                    │ │    │
│  │  │    - OLS vs Kalman vs Robust                 │ │    │
│  │  │  • Spread charts & z-score                   │ │    │
│  │  │  • Rolling correlation heatmap               │ │    │
│  │  │  • Statistical tests (ADF, cointegration)    │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │  Tab 3: Backtesting                          │ │    │
│  │  │  • Parameter configuration                   │ │    │
│  │  │    - Entry/exit z-scores                     │ │    │
│  │  │    - Stop loss thresholds                    │ │    │
│  │  │  • Run backtest button                       │ │    │
│  │  │  • Performance metrics table                 │ │    │
│  │  │    - Total return, Sharpe, Drawdown          │ │    │
│  │  │  • Equity curve chart                        │ │    │
│  │  │  • Trade history table                       │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │  Tab 4: Alerts & Monitoring                  │ │    │
│  │  │  • Create alert form                         │ │    │
│  │  │    - Alert type (price, z-score, volume)     │ │    │
│  │  │    - Threshold and condition                 │ │    │
│  │  │  • Active alerts list                        │ │    │
│  │  │  • Triggered alerts feed                     │ │    │
│  │  │    - Real-time updates                       │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │  Tab 5: Data Management                      │ │    │
│  │  │  • Export data (CSV)                         │ │    │
│  │  │    - Date range selector                     │ │    │
│  │  │    - Timeframe selector                      │ │    │
│  │  │  • Upload OHLC data                          │ │    │
│  │  │    - CSV/JSON upload                         │ │    │
│  │  │  • System statistics                         │ │    │
│  │  │    - Tick count, symbols, uptime             │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────── │    │
│                                                              │
│  Interactive Features:                                       │
│  • Auto-refresh every 2 seconds                             │
│  • Zoom, pan, hover on all charts                           │
│  • Responsive layout                                         │
│  • Custom CSS styling                                        │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────┐
│ Binance  │
│ WebSocket│
└────┬─────┘
     │ (1) Tick Data Stream
     │     {symbol, ts, price, size}
     ▼
┌────────────────────┐
│ WebSocket Client   │ ──────┐
│ - Parse & validate │       │ (2) Callback
│ - Reconnect logic  │       │     invoke
└────────────────────┘       │
                             ▼
                    ┌─────────────────────┐
                    │ Data Processor      │
                    │ - Process tick      │
                    │ - Update stats      │
                    │ - Check alerts      │
                    └──┬────────────────┬─┘
                       │                │
          (3) Store   │                │ (4) Cache
                       │                │
                       ▼                ▼
              ┌─────────────┐  ┌──────────────┐
              │ PostgreSQL  │  │    Redis     │
              │ - Tick data │  │ - Latest     │
              │ - OHLCV     │  │ - Stats      │
              │ - Resampled │  │ - Alerts     │
              └──────┬──────┘  └──────┬───────┘
                     │                 │
                     │ (5) Query       │ (6) Read
                     │                 │
                     ▼                 ▼
              ┌──────────────────────────────┐
              │   FastAPI REST API           │
              │   - GET /api/price           │
              │   - GET /api/ohlc            │
              │   - GET /api/analytics       │
              │   - POST /api/backtest       │
              └───────────┬──────────────────┘
                          │
                          │ (7) HTTP Requests
                          │     JSON Responses
                          ▼
              ┌──────────────────────────────┐
              │  Streamlit Frontend          │
              │  - Fetch data                │
              │  - Render charts             │
              │  - User interactions         │
              └──────────────────────────────┘
```

## Component Communication

### 1. WebSocket Data Ingestion
- **Protocol**: WSS (WebSocket Secure)
- **Frequency**: Real-time (multiple updates/second)
- **Data Format**: JSON trade events
- **Error Handling**: Auto-reconnect with exponential backoff

### 2. Backend Processing
- **Pattern**: Async event-driven
- **Concurrency**: asyncio event loop
- **Callback Chain**: WebSocket → Processor → Storage → Cache

### 3. Database Operations
- **PostgreSQL**: Async writes via asyncpg
- **Batch Inserts**: Bulk operations for performance
- **Indexing**: Timestamp + symbol for fast queries

### 4. Caching Strategy
- **Redis**: Sub-millisecond reads
- **TTL Policy**: Expire old data automatically
- **Pub/Sub**: Alert notifications

### 5. API Layer
- **REST**: Synchronous requests for data
- **WebSocket**: Real-time push for updates (optional)
- **Rate Limiting**: Can be added for production

### 6. Frontend Updates
- **Polling**: Every 2 seconds
- **Lazy Loading**: Only fetch visible data
- **Caching**: Browser-side caching for static data

## Scalability Considerations

### Horizontal Scaling
```
                    ┌──────────────┐
                    │ Load Balancer│
                    │  (Nginx)     │
                    └───────┬──────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                   ▼
    ┌──────────┐      ┌──────────┐       ┌──────────┐
    │ Backend 1│      │ Backend 2│       │ Backend 3│
    │ (FastAPI)│      │ (FastAPI)│       │ (FastAPI)│
    └─────┬────┘      └─────┬────┘       └─────┬────┘
          │                 │                   │
          └─────────────────┼───────────────────┘
                            │
                    ┌───────▼──────┐
                    │ PostgreSQL   │
                    │ (Primary +   │
                    │  Replicas)   │
                    └──────────────┘
```

### Vertical Scaling
- Increase PostgreSQL `shared_buffers`
- Add Redis replicas for read scaling
- Use connection pooling (pgbouncer)
- Add indexes for common queries

### Data Partitioning
- Partition tick_data by symbol or date
- Use TimescaleDB for automatic partitioning
- Archive old data to cold storage

## Security Considerations

### Current Implementation (Development)
- No authentication (local deployment)
- CORS enabled for all origins
- No HTTPS/TLS (localhost)

### Production Recommendations
- **API Auth**: JWT tokens or API keys
- **Database**: SSL connections, strong passwords
- **HTTPS**: TLS certificates for all endpoints
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize all inputs
- **Secrets Management**: Use environment variables

## Monitoring & Logging

### Current Logging
- Python logging module
- Logs to stdout (Docker logs)
- Error tracking in exceptions

### Production Recommendations
- **APM**: Application Performance Monitoring (e.g., DataDog)
- **Log Aggregation**: ELK stack or Splunk
- **Metrics**: Prometheus + Grafana
- **Alerts**: PagerDuty for critical issues
- **Health Checks**: /api/health endpoint

## Deployment Architecture

### Development (Current)
```
Docker Compose on Single Host
- All services in one docker-compose.yml
- Data volumes for persistence
- Internal Docker network
```

### Production (Recommended)
```
Kubernetes Cluster
- Separate pods for each service
- Auto-scaling based on load
- Persistent volumes for databases
- Ingress for load balancing
- ConfigMaps for configuration
```

## Technology Decisions Rationale

| Component | Choice | Rationale | Alternative |
|-----------|--------|-----------|-------------|
| Backend | FastAPI | Modern, async, fast, auto-docs | Flask, Django |
| Database | PostgreSQL | Rich queries, ACID, time-series support | TimescaleDB, InfluxDB |
| Cache | Redis | Sub-ms latency, pub/sub, simple | Memcached |
| Frontend | Streamlit | Rapid development, Python-native | Dash, React |
| Container | Docker | Standard, portable, easy | Podman |
| Orchestration | Docker Compose | Simple for single-host | Kubernetes |

---

This architecture supports:
- ✅ Real-time data ingestion
- ✅ Multi-timeframe analysis
- ✅ Advanced analytics
- ✅ Interactive visualization
- ✅ Alerting system
- ✅ Data export/import
- ✅ Extensibility
- ✅ Scalability (with modifications)
