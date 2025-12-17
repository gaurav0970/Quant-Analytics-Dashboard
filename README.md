# Quant Analytics Platform

A real-time quantitative analytics platform for cryptocurrency trading.

The system ingests live Binance WebSocket data, performs statistical arbitrage and pair trading analytics, and presents results through an interactive dashboard.

Built for quant traders, researchers, and low-latency analytics workflows.

## Overview

This platform provides an end-to-end pipeline:

- Real-time tick ingestion
- Statistical and econometric analysis
- Mean reversion strategy evaluation
- Interactive visualization and alerts

The system is modular, production-oriented, and Docker-ready.

## Key Features

### Real-Time Data

- Live Binance WebSocket tick data
- Multi-symbol support
- Sub-second ingestion latency

### Quant Analytics

**Hedge ratio estimation:**
- OLS
- Kalman Filter (time-varying)
- Robust regression (Huber, Theil-Sen)

**Statistical analysis:**
- Spread and Z-score computation
- ADF stationarity test
- Cointegration analysis
- Rolling correlation

### Pair Trading and Backtesting

- Mean reversion strategy logic
- Configurable entry and exit thresholds
- Performance metrics and trade statistics

### Dashboard

- Streamlit-based user interface
- Plotly interactive charts (zoom, pan, hover)
- Symbol, timeframe, and rolling window controls
- CSV data export

### Alerts

- Price-based alerts
- Z-score deviation alerts
- Redis-backed real-time triggering

## Architecture

### System Architecture Overview

![Quant Analytics Platform Architecture](assets/architecture.png)

### Live Data Flow and Runtime Behavior

#### Live Data Flow

Once the system is running, it continuously performs the following operations:

1. Receives real-time tick data from Binance WebSocket streams
2. Stores raw tick data in PostgreSQL for historical analysis
3. Caches the latest prices and metrics in Redis for low-latency access
4. Resamples tick data into multiple timeframes (1s, 1m, 5m)
5. Calculates analytics and updates dashboard visualizations in real time

This pipeline operates asynchronously and is designed for low-latency, high-throughput data processing.

#### Advanced Analytics (After Data Accumulation)

After approximately 30–60 minutes of live data collection, sufficient data is available to enable deeper analysis:

- Run mean reversion backtests on selected pairs
- Analyze spread behavior and rolling Z-scores
- Compare hedge ratio estimation methods (OLS, Kalman Filter, Robust regression)
- Evaluate stationarity using ADF and cointegration tests
- Export processed data and analytics results as CSV files

These features are intentionally data-dependent to ensure statistical validity.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, SQLAlchemy, asyncpg |
| Storage | PostgreSQL |
| Caching | Redis |
| Frontend | Streamlit, Plotly |
| Analytics | NumPy, SciPy, Statsmodels, Scikit-learn, PyKalman |
| Infrastructure | Docker, Docker Compose, Uvicorn |

## Prerequisites

### Recommended

- Docker Desktop or Docker Engine
- Minimum 4 GB RAM

### Local Development

- Python 3.8+
- PostgreSQL 13+
- Redis 6.0+

## Quick Start

### Docker Setup (Recommended)

```bash
docker-compose up --build
```

**Access:**
- Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs

**Stop services:**
```bash
docker-compose down
```

### Local Development Setup

#### 1. Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Database Setup

```bash
psql -U postgres
CREATE DATABASE quantdb;
CREATE USER quantuser WITH PASSWORD 'quantpass';
GRANT ALL PRIVILEGES ON DATABASE quantdb TO quantuser;
\q
```

#### 3. Environment Variables (.env)

```bash
DATABASE_URL=postgresql://quantuser:quantpass@localhost:5432/quantdb
REDIS_URL=redis://localhost:6379/0
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT
LOG_LEVEL=INFO
```

#### 4. Run Services

**Backend:**
```bash
cd backend
uvicorn app:app --port 8000 --reload
```

**Frontend (in another terminal):**
```bash
cd frontend
streamlit run app.py --server.port 8501
```

**Access:** http://localhost:8501

## Project Structure

```
backend/
├── app.py                  # FastAPI application
├── data_processor.py       # Tick ingestion and resampling
├── analytics/              # OLS, Kalman, regression, tests
└── models/                 # Database models

frontend/
└── app.py                  # Streamlit dashboard

docker-compose.yml
Dockerfile
requirements.txt
```

## Analytics Details

### Hedge Ratio Estimation

- **OLS**: Fast baseline method
- **Kalman Filter**: Adaptive, time-varying relationships
- **Robust regression**: Outlier-resistant estimation

### Strategy Rules

```
Entry: |z_score| > 2.0
Exit:  |z_score| < 0.5
Stop:  |z_score| > 4.0
```

## API Endpoints

### Price Statistics

```bash
GET /api/stats
```

### Pair Analysis

```bash
POST /api/analyze
{
  "symbol1": "BTCUSDT",
  "symbol2": "ETHUSDT"
}
```

### Backtest

```bash
POST /api/backtest
{
  "symbol1": "BTCUSDT",
  "symbol2": "ETHUSDT"
}
```

**Full API Documentation:** http://localhost:8000/docs

## 🎥 Video Demonstration

### GitHub Video Upload (Recommended)

Create a directory in the repository:
```
/assets/demo/
```

Upload your video file:
```
/assets/demo/platform_walkthrough.mp4
```

Add the following line to embed the video:

```markdown
[![Platform Walkthrough](https://github.com/<your-username>/<repo-name>/assets/demo/platform_walkthrough.mp4)](https://github.com/<your-username>/<repo-name>/assets/demo/platform_walkthrough.mp4)
```

GitHub will automatically render the video inline.

**Video Contents (25 min):**
- Architecture overview
- WebSocket data ingestion
- Price monitor dashboard
- Pair trading analytics
- Statistical tests
- Mean reversion backtest
- Alert system
- Data export

## Troubleshooting

### Backend Logs

```bash
docker-compose logs backend
```

### No Data Ingestion

```bash
curl http://localhost:8000/api/stats
```

### Port Conflicts

```bash
# Windows
netstat -ano | findstr :8000

# Mac/Linux
lsof -i :8000
```

### Common Issues

- **Port 8000 in use**: Kill the process or use a different port
- **PostgreSQL not ready**: Wait 10 seconds after docker-compose starts
- **Redis connection failed**: Verify Redis is running and accessible



