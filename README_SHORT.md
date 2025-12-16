# Quant Analytics Platform

Complete real-time cryptocurrency trading analytics. Ingests Binance WebSocket data, performs statistical arbitrage analytics, provides interactive dashboard.

## 🎯 Key Features

- **Real-time Data**: Live Binance WebSocket tick data
- **Analytics**: OLS, Kalman Filter, Robust regression, ADF test, cointegration
- **Dashboard**: Streamlit UI with Plotly charts (zoom, pan, hover)
- **Pair Trading**: Spread analysis, z-scores, mean reversion backtesting
- **Alerting**: Custom price/z-score alerts
- **Data Export**: Download analytics as CSV

## 🏗️ Tech Stack

FastAPI, PostgreSQL, Redis, Streamlit, Plotly, NumPy, SciPy, Statsmodels, PyKalman

## 📋 Prerequisites

- **Docker**: Docker Desktop/Engine, 4GB RAM (Recommended)
- **Local**: Python 3.8+, PostgreSQL 13+, Redis 6.0+

## 🚀 Quick Start

### Docker (5 min)
```bash
cd "C:\Users\gaura\Downloads\try 3"
docker-compose up --build
```
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Local Setup (20 min)

**1. Install Dependencies**
```bash
# Windows: Download Python & PostgreSQL
# Mac: brew install python@3.10 postgresql redis
# Linux: sudo apt install python3.10 postgresql redis-server

cd "C:\Users\gaura\Downloads\try 3"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Setup Database**
```bash
psql -U postgres
CREATE DATABASE quantdb;
CREATE USER quantuser WITH PASSWORD 'quantpass';
GRANT ALL PRIVILEGES ON DATABASE quantdb TO quantuser;
\q
```

**3. Configure .env**
```bash
DATABASE_URL=postgresql://quantuser:quantpass@localhost:5432/quantdb
REDIS_URL=redis://localhost:6379/0
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT
LOG_LEVEL=INFO
```

**4. Run Services**
```bash
# Terminal 1 - Backend
cd backend
python -m uvicorn app:app --port 8000 --reload

# Terminal 2 - Frontend
cd frontend
streamlit run app.py --server.port 8501
```

Access: http://localhost:8501

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | REST API |
| streamlit | 1.28.1 | Dashboard |
| postgresql | 13+ | Database |
| redis | 7.0+ | Cache |
| pandas | 2.1.2 | Data manipulation |
| numpy | 1.26.3 | Numerics |
| scipy | 1.11.4 | Statistics |
| scikit-learn | 1.3.2 | Regression |
| statsmodels | 0.14.0 | ADF, cointegration |
| pykalman | 0.9.5 | Kalman filter |

## 🎮 Dashboard

- **Price Monitor**: Real-time charts
- **Pair Analytics**: Spread, z-scores
- **Statistical Tests**: ADF, cointegration
- **Backtesting**: Strategy results
- **Alerts**: Custom triggers
- **Data Export**: CSV downloads

## 📦 Project Structure

```
backend/
├── app.py                  # FastAPI app
├── data_processor.py       # Tick processing
├── analytics/              # OLS, Kalman, regression
└── models/                 # Database models

frontend/
└── app.py                  # Streamlit dashboard

docker-compose.yml         # Service orchestration
requirements.txt           # Dependencies
```

## 🎥 Video Demonstration

**[📹 Platform Walkthrough (25 min)](https://youtube.com/your-video-link)**

- Architecture overview (2 min)
- WebSocket data ingestion (3 min)
- Price monitor dashboard (4 min)
- Pair trading analytics (5 min)
- Statistical tests (3 min)
- Mean reversion backtest (4 min)
- Alert system (2 min)
- Data export (2 min)

**Setup Video (7 min)**
- Docker installation
- Database configuration
- First run verification

## 📊 Analytics Methods

**Hedge Ratios**: OLS (fast), Kalman (adaptive), Robust (outlier resistant)

**Strategy Rules**:
```
Entry: |z_score| > 2.0
Exit: |z_score| < 0.5
Stop Loss: |z_score| > 4.0
```

## 🚀 API Endpoints

```bash
# Price statistics
curl http://localhost:8000/api/stats

# Analyze pair
curl -X POST http://localhost:8000/api/analyze \
  -d '{"symbol1":"BTCUSDT", "symbol2":"ETHUSDT"}'

# Run backtest
curl -X POST http://localhost:8000/api/backtest \
  -d '{"symbol1":"BTCUSDT", "symbol2":"ETHUSDT"}'

# Docs: http://localhost:8000/docs
```

## 🔧 Configuration

```bash
# Add new symbol to .env
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT

# Restart backend
docker-compose restart backend
```

## 🐛 Troubleshooting

**Backend won't start:**
```bash
docker-compose logs backend
```

**No data in dashboard:**
```bash
curl http://localhost:8000/api/stats
```

**Port in use:**
```bash
netstat -ano | findstr :8000    # Windows
lsof -i :8000                   # Mac/Linux
```

## 📈 Performance

- Tick ingestion: ~1000 ticks/second
- Storage: ~500MB per day
- API response: <500ms
- Dashboard refresh: 1-2s

## 🔗 Quick Commands

```bash
# Docker
docker-compose up --build
docker-compose down
docker-compose logs -f backend
docker stats

# Local
python -m venv venv
pip install -r requirements.txt
psql -U quantuser -d quantdb
redis-cli ping
```

## 📚 References

- [Binance WebSocket](https://binance-docs.github.io/apidocs/futures/en/)
- [Pairs Trading](https://www.investopedia.com/terms/p/pairstrade.asp)
- [Kalman Filter](https://www.kalmanfilter.net/)
- [ADF Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)

## 📄 License

Educational and evaluation purposes.

## 👤 Author

Quant Developer evaluation assignment.

## 🙏 Acknowledgments

- ChatGPT/Claude for code assistance
- Binance for WebSocket API
- Open-source community
