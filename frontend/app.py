import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import asyncio
import websockets
import threading
from typing import Dict, List, Optional
import io
import os

# Page configuration
st.set_page_config(
    page_title="Quant Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")  # Changed from localhost
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPSDT"]
TIMEFRAMES = ["1s", "1m", "5m"]
ANALYTICS_TYPES = ["Price Stats", "Pair Analytics", "Mean Reversion", "Statistical Tests"]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-triggered {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-normal {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        self.backend_url = BACKEND_URL
        # For WebSocket, use the service name when in Docker, localhost when local
        ws_host = "backend" if os.getenv("BACKEND_URL") else "localhost"
        self.websocket_url = f"ws://{ws_host}:8000/ws/updates"
        self.websocket_thread = None
        self.running = False
        
        # Log the backend URL for debugging
        print(f"Using backend URL: {self.backend_url}")
        print(f"Using WebSocket URL: {self.websocket_url}")
        
    def fetch_data(self, endpoint: str, params: dict = None):
        """Fetch data from backend API"""
        try:
            url = f"{self.backend_url}{endpoint}"
            print(f"Fetching from: {url}")  # Debug log
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            st.error(f"Cannot connect to backend at {self.backend_url}. Make sure backend is running.")
            print(f"Connection error: {e}")  # Debug log
            return None
        except Exception as e:
            st.error(f"Error fetching data from {endpoint}: {e}")
            print(f"Error details: {e}")  # Debug log
            return None
    
    def post_data(self, endpoint: str, data: dict):
        """Post data to backend API"""
        try:
            url = f"{self.backend_url}{endpoint}"
            print(f"Posting to: {url}")  # Debug log
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            st.error(f"Cannot connect to backend at {self.backend_url}")
            print(f"Connection error: {e}")
            return None
        except Exception as e:
            st.error(f"Error posting data to {endpoint}: {e}")
            print(f"Error details: {e}")
            return None
    
    # ... rest of the Dashboard class methods remain the same ...
    
    def start_websocket(self):
        """Start WebSocket connection in background thread"""
        def websocket_loop():
            asyncio.run(self._websocket_handler())
        
        self.websocket_thread = threading.Thread(target=websocket_loop, daemon=True)
        self.websocket_thread.start()
    
    async def _websocket_handler(self):
        """Handle WebSocket messages"""
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    while self.running:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            # Update session state with new data
                            if 'type' in data and 'data' in data:
                                self._handle_websocket_message(data['type'], data['data'])
                        except websockets.exceptions.ConnectionClosed:
                            break
                        except Exception as e:
                            st.error(f"WebSocket error: {e}")
                            break
            except Exception as e:
                st.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)
    
    def _handle_websocket_message(self, message_type: str, data: dict):
        """Handle incoming WebSocket messages"""
        if message_type == "price_update":
            symbol = data.get('symbol')
            price = data.get('price')
            if symbol and price:
                # Update session state
                if 'latest_prices' not in st.session_state:
                    st.session_state.latest_prices = {}
                st.session_state.latest_prices[symbol] = price
                
                # Trigger rerun if this symbol is being displayed
                if st.session_state.get('selected_symbol') == symbol:
                    st.rerun()
    
    def create_price_chart(self, ohlc_data: List[dict], symbol: str):
        """Create OHLC chart"""
        if not ohlc_data:
            return None
        
        df = pd.DataFrame(ohlc_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Add VWAP line
        if 'vwap' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['vwap'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color='blue',
                opacity=0.5
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark"
        )
        
        return fig
    
    def create_spread_chart(self, spread_data: List[float], zscore_data: List[float], 
                           timestamps: List[str]):
        """Create spread and z-score chart"""
        if not spread_data or not zscore_data:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Spread", "Z-Score")
        )
        
        # Spread chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=spread_data,
                mode='lines',
                name='Spread',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=1
        )
        
        # Add mean line
        spread_mean = np.mean(spread_data) if spread_data else 0
        fig.add_hline(
            y=spread_mean,
            line_dash="dash",
            line_color="gray",
            row=1, col=1
        )
        
        # Z-score chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=zscore_data,
                mode='lines',
                name='Z-Score',
                line=dict(color='magenta', width=2)
            ),
            row=2, col=1
        )
        
        # Add z-score thresholds
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame):
        """Create correlation heatmap"""
        if correlation_matrix.empty:
            return None
        
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            title="Rolling Correlation Matrix",
            height=500,
            template="plotly_dark"
        )
        
        return fig
    
    def create_metrics_display(self, metrics: dict):
        """Create metrics display"""
        cols = st.columns(4)
        
        for idx, (key, value) in enumerate(metrics.items()):
            with cols[idx % 4]:
                st.metric(
                    label=key.replace('_', ' ').title(),
                    value=f"{value:,.4f}" if isinstance(value, (int, float)) else value
                )

def main():
    st.markdown("<h1 class='main-header'>📊 Quant Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = Dashboard()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Symbol",
            SYMBOLS,
            index=0
        )
        
        # Timeframe selection
        selected_timeframe = st.selectbox(
            "Select Timeframe",
            TIMEFRAMES,
            index=1
        )
        
        # Analytics type selection
        selected_analytics = st.selectbox(
            "Select Analytics Type",
            ANALYTICS_TYPES,
            index=0
        )
        
        # Pair selection for pair analytics
        if selected_analytics == "Pair Analytics":
            col1, col2 = st.columns(2)
            with col1:
                pair_symbol1 = st.selectbox(
                    "Symbol 1",
                    SYMBOLS,
                    index=0
                )
            with col2:
                pair_symbol2 = st.selectbox(
                    "Symbol 2",
                    SYMBOLS,
                    index=1
                )
        
        # Alert configuration
        st.subheader("Alert Configuration")
        
        alert_symbol = st.selectbox(
            "Alert Symbol",
            SYMBOLS,
            key="alert_symbol"
        )
        
        alert_type = st.selectbox(
            "Alert Type",
            ["price", "zscore", "volume"],
            key="alert_type"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            alert_condition = st.selectbox(
                "Condition",
                [">", "<", "=="],
                key="alert_condition"
            )
        with col2:
            alert_threshold = st.number_input(
                "Threshold",
                value=0.0,
                step=0.1,
                key="alert_threshold"
            )
        
        if st.button("Create Alert", type="primary"):
            alert_data = {
                "symbol": alert_symbol,
                "type": alert_type,
                "condition": alert_condition,
                "threshold": alert_threshold,
                "message": f"{alert_symbol} {alert_type} {alert_condition} {alert_threshold}"
            }
            
            result = dashboard.post_data("/api/alerts", alert_data)
            if result:
                st.success("Alert created successfully!")
        
        # Data export
        st.subheader("Data Export")
        
        export_symbol = st.selectbox(
            "Export Symbol",
            SYMBOLS,
            key="export_symbol"
        )
        
        export_timeframe = st.selectbox(
            "Timeframe",
            ["tick"] + TIMEFRAMES,
            key="export_timeframe"
        )
        
        if st.button("Export Data", type="secondary"):
            params = {
                "symbol": export_symbol,
                "timeframe": export_timeframe if export_timeframe != "tick" else None
            }
            
            result = dashboard.fetch_data(f"/api/export/{export_symbol}", params)
            if result:
                # Create download link
                csv_data = result['data']
                filename = result['filename']
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Analytics", "Alerts", "System Info"])
    
    with tab1:
        # Real-time metrics
        st.subheader("Real-time Metrics")
        
        # Fetch current price
        price_data = dashboard.fetch_data(f"/api/price/{selected_symbol}")
        
        if price_data:
            cols = st.columns(4)
            with cols[0]:
                st.metric(
                    label=f"{selected_symbol} Price",
                    value=f"${price_data['price']:,.2f}",
                    delta=None
                )
            with cols[1]:
                st.metric(
                    label="Timestamp",
                    value=price_data['timestamp'][11:19]  # Just time
                )
            with cols[2]:
                st.metric(
                    label="Size",
                    value=f"{price_data['size']:,.4f}"
                )
        
        # Price chart
        st.subheader("Price Chart")
        
        ohlc_data = dashboard.fetch_data(
            f"/api/ohlc/{selected_symbol}",
            {"timeframe": selected_timeframe, "limit": 100}
        )
        
        if ohlc_data and 'data' in ohlc_data:
            fig = dashboard.create_price_chart(ohlc_data['data'], selected_symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available yet. Waiting for data...")
    
    with tab2:
        if selected_analytics == "Price Stats":
            st.subheader("Price Statistics")
            
            analytics_data = dashboard.fetch_data(
                f"/api/analytics/symbol/{selected_symbol}",
                {"timeframe": selected_timeframe}
            )
            
            if analytics_data and 'analytics' in analytics_data:
                analytics = analytics_data['analytics']
                
                # Display metrics
                if 'price_stats' in analytics:
                    dashboard.create_metrics_display(analytics['price_stats'])
                
                # Additional charts
                if 'rolling_stats' in analytics:
                    st.subheader("Rolling Statistics")
                    
                    rolling_df = pd.DataFrame(analytics['rolling_stats'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=rolling_df.get('rolling_mean', []),
                        mode='lines',
                        name='Rolling Mean',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=rolling_df.get('bollinger_upper', []),
                        mode='lines',
                        name='Bollinger Upper',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        y=rolling_df.get('bollinger_lower', []),
                        mode='lines',
                        name='Bollinger Lower',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Bollinger Bands",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif selected_analytics == "Pair Analytics":
            st.subheader(f"Pair Analytics: {pair_symbol1} / {pair_symbol2}")
            
            analytics_data = dashboard.fetch_data(
                f"/api/analytics/pair/{pair_symbol1}/{pair_symbol2}",
                {"timeframe": selected_timeframe}
            )
            
            if analytics_data and 'analytics' in analytics_data:
                analytics = analytics_data['analytics']
                
                # Display hedge ratio and z-score
                cols = st.columns(4)
                if 'ols' in analytics:
                    with cols[0]:
                        st.metric(
                            "OLS Hedge Ratio",
                            f"{analytics['ols'].get('hedge_ratio', 0):.4f}"
                        )
                    with cols[1]:
                        st.metric(
                            "R²",
                            f"{analytics['ols'].get('r_squared', 0):.4f}"
                        )
                
                if 'spread' in analytics:
                    with cols[2]:
                        st.metric(
                            "Current Z-Score",
                            f"{analytics['spread'].get('current_zscore', 0):.2f}"
                        )
                    with cols[3]:
                        signal = analytics['spread'].get('signal', 0)
                        signal_text = "Long" if signal == 1 else "Short" if signal == -1 else "Flat"
                        st.metric("Signal", signal_text)
                
                # Spread and z-score chart
                if 'spread' in analytics:
                    spread_data = analytics['spread'].get('spread', [])
                    zscore_data = analytics['spread'].get('zscore', [])
                    
                    if spread_data and zscore_data:
                        # Create timestamps
                        timestamps = [f"T-{i}" for i in range(len(spread_data)-1, -1, -1)]
                        
                        fig = dashboard.create_spread_chart(spread_data, zscore_data, timestamps)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                # Regression results
                if 'ols' in analytics:
                    with st.expander("OLS Regression Details"):
                        st.json(analytics['ols'])
                
                # Statistical tests
                if 'adf_test' in analytics:
                    with st.expander("ADF Test Results"):
                        st.json(analytics['adf_test'])
                
                if 'cointegration' in analytics:
                    with st.expander("Cointegration Test Results"):
                        st.json(analytics['cointegration'])
        
        elif selected_analytics == "Mean Reversion":
            st.subheader("Mean Reversion Analytics")
            
            # Select pair for mean reversion
            col1, col2 = st.columns(2)
            with col1:
                mr_symbol1 = st.selectbox("Symbol 1", SYMBOLS, key="mr_sym1")
            with col2:
                mr_symbol2 = st.selectbox("Symbol 2", SYMBOLS, key="mr_sym2", index=1)
            
            # Backtest parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                entry_z = st.number_input("Entry Z-Score", value=2.0, step=0.1)
            with col2:
                exit_z = st.number_input("Exit Z-Score", value=0.0, step=0.1)
            with col3:
                capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
            
            if st.button("Run Backtest", type="primary"):
                # This would call a backtest endpoint
                st.info("Backtest functionality would be implemented with historical data")
        
        elif selected_analytics == "Statistical Tests":
            st.subheader("Statistical Tests")
            
            analytics_data = dashboard.fetch_data(
                f"/api/analytics/symbol/{selected_symbol}",
                {"timeframe": selected_timeframe}
            )
            
            if analytics_data and 'analytics' in analytics_data:
                analytics = analytics_data['analytics']
                
                if 'hurst_exponent' in analytics:
                    hurst = analytics['hurst_exponent']
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Hurst Exponent", f"{hurst:.4f}")
                    
                    # Interpretation
                    if hurst > 0.5:
                        interpretation = "Trending (Persistent)"
                        color = "green"
                    elif hurst < 0.5:
                        interpretation = "Mean-Reverting (Anti-Persistent)"
                        color = "blue"
                    else:
                        interpretation = "Random Walk"
                        color = "gray"
                    
                    with cols[1]:
                        st.markdown(f"<p style='color:{color}; font-weight:bold;'>{interpretation}</p>", 
                                   unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Alert Management")
        
        # Triggered alerts
        st.subheader("Triggered Alerts")
        
        alerts_data = dashboard.fetch_data("/api/alerts/triggered", {"limit": 10})
        
        if alerts_data and 'alerts' in alerts_data:
            for alert in alerts_data['alerts']:
                with st.container():
                    condition = alert.get('condition', '>')
                    current_val = alert.get('current_value', 0)
                    threshold = alert.get('threshold', 0)
                    timestamp = alert.get('timestamp', '')
                    time_str = timestamp[11:19] if len(timestamp) > 11 else timestamp
                    
                    st.markdown(f"""
                    <div class='alert-triggered'>
                        <strong>{alert['symbol']}</strong> - {alert['alert_type']} {condition} {threshold}<br>
                        Current: {current_val:.4f} | Time: {time_str}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No alerts triggered yet")
        
        # Active alerts (would require additional endpoint)
        st.subheader("Configured Alerts")
        st.info("Alert configuration is in the sidebar")
    
    with tab4:
        st.subheader("System Information")
        
        # Health check
        health_data = dashboard.fetch_data("/api/health")
        
        if health_data:
            cols = st.columns(3)
            with cols[0]:
                status = health_data['status']
                color = "green" if status == "healthy" else "orange"
                st.markdown(f"<p style='color:{color}; font-weight:bold;'>Status: {status}</p>", 
                           unsafe_allow_html=True)
            
            with cols[1]:
                ws_status = health_data['websocket']
                color = "green" if ws_status == "connected" else "red"
                st.markdown(f"<p style='color:{color}; font-weight:bold;'>WebSocket: {ws_status}</p>", 
                           unsafe_allow_html=True)
            
            with cols[2]:
                st.metric("Timestamp", health_data['timestamp'][11:19])
        
        # System stats
        stats_data = dashboard.fetch_data("/api/stats")
        
        if stats_data:
            st.subheader("Database Statistics")
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Ticks", stats_data.get('tick_count', 0))
            
            with cols[1]:
                active_symbols = len(stats_data.get('active_symbols', []))
                st.metric("Active Symbols", active_symbols)
            
            with cols[2]:
                redis_status = "Connected" if stats_data.get('redis_connected') else "Disconnected"
                st.metric("Redis", redis_status)
            
            with cols[3]:
                ws_running = "Running" if stats_data.get('websocket_running') else "Stopped"
                st.metric("WebSocket", ws_running)
            
            # Symbol counts
            st.subheader("Symbol Statistics")
            if 'symbol_counts' in stats_data:
                symbol_df = pd.DataFrame(
                    list(stats_data['symbol_counts'].items()),
                    columns=['Symbol', 'Tick Count']
                )
                st.dataframe(symbol_df, use_container_width=True)
            
            # Latest ticks
            st.subheader("Latest Ticks")
            if 'latest_ticks' in stats_data:
                latest_df = pd.DataFrame(stats_data['latest_ticks'])
                st.dataframe(latest_df, use_container_width=True)

if __name__ == "__main__":
    main()