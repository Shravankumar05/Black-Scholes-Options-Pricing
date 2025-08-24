# Import deployment configuration FIRST to configure TensorFlow before any ML imports
try:
    from deployment_config import deployment_config, safe_pyplot_display, configure_deployment_environment, safe_yfinance_download
    # Configure environment immediately
    if deployment_config.get('is_deployed', False):
        configure_deployment_environment()
    DEPLOYMENT_CONFIG_AVAILABLE = True
except ImportError:
    DEPLOYMENT_CONFIG_AVAILABLE = False
    def safe_pyplot_display(fig, **kwargs):
        import matplotlib.pyplot as plt
        import streamlit as st
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    
    def safe_yfinance_download(symbol, **kwargs):
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.history(**kwargs)

import streamlit as st
import numpy as np
import matplotlib
import os
import yfinance as yf
import matplotlib.pyplot as plt

from bs_functions import black_scholes_call, black_scholes_puts, implied_volatility, delta_call, delta_put, gamma, theta_put, theta_call, vega, phi, pdf, black_scholes_vectorized, black_scholes_multithreaded, black_scholes_optimized, black_scholes_jit, phi_vectorized, pdf_vectorized, NUMBA_AVAILABLE
from db_utils import create_table, insert_calculation, insert_output, insert_outputs_bulk
from portfolio_risk import render_portfolio_risk_page
st.set_page_config(page_title="Black-Scholes Options Dashboard", page_icon="chart_with_upwards_trend",  layout="wide", initial_sidebar_state="expanded")

try:
    from options_market_data import options_market_data
    from enhanced_visualizations import enhanced_viz
    from ml_components import ml_predictor, vol_forecaster, regime_detector
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    st.error(f"Enhanced features not available: {e}")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #212529 0%, #343a40 100%);
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stTextInput label,
    .stSidebar .stNumberInput label,
    .stSidebar .stSlider label {
        color: #f8f9fa !important;
        font-weight: 600;
    }
    
    .stSidebar .stMarkdown {
        color: #f8f9fa;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #f8f9fa !important;
    }
    
    /* Professional input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid #212529 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #495057 !important;
        box-shadow: 0 0 0 3px rgba(73, 80, 87, 0.1) !important;
    }
    
    .stSlider > div > div > div {
        border: 1px solid #212529 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #212529;
        color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #495057 !important;
        color: #f8f9fa !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #212529 0%, #495057 100%);
        color: #f8f9fa;
        border: 2px solid #212529;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #495057 0%, #6c757d 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #b8dabd;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-banner {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #abdde5;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    h1, h2, h3 {
        color: #212529 !important;
        font-weight: 700 !important;
    }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    .explanation-box {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-left: 4px solid #495057;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

analysis_modes = ["Individual Black-Scholes", "Portfolio Risk Analysis"]
if ENHANCED_FEATURES_AVAILABLE:
    analysis_modes.extend(["Options Market Analysis", "ML Volatility Forecasting"])

page = st.sidebar.selectbox("Select Analysis Type", analysis_modes, index=0)
if page == "Portfolio Risk Analysis":
    render_portfolio_risk_page()
    st.stop()
elif page == "Options Market Analysis" and ENHANCED_FEATURES_AVAILABLE:
    st.title("Live Options Market Analysis")
    
    st.markdown("""
    <div class="explanation-box">
    <h4>Live Options Market Analysis</h4>
    <p>This section provides real-time options market data analysis including:</p>
    <ul>
        <li>Live options chain data for any stock symbol</li>
        <li>Implied volatility surface visualization</li>
        <li>Options flow analysis to identify unusual activity</li>
        <li>Machine learning models trained on market data</li>
    </ul>
    <p><strong>How to use:</strong> Enter a stock symbol, select your target days to expiry, and click "Fetch Options Data" to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Market Data Controls")
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, MSFT, TSLA)")
        days_to_expiry = st.slider("Target Days to Expiry", 7, 90, 30)
        
        if st.button("Fetch Options Data", type="primary", use_container_width=True):
            with st.spinner("Fetching options market data..."):
                options_data = options_market_data.fetch_options_chain(symbol, days_to_expiry)
                
                if options_data:
                    st.session_state['options_data'] = options_data
                    st.success(f"Loaded options data for {symbol}")
                else:
                    st.error("Failed to fetch options data")
        
        if st.button("Analyze Options Flow", use_container_width=True):
            with st.spinner("Analyzing options flow..."):
                flow_data = options_market_data.analyze_options_flow(symbol)
                
                if flow_data:
                    st.session_state['flow_data'] = flow_data
                    st.success("Options flow analysis complete")
                else:
                    st.error("Failed to analyze options flow")
    
    with col2:
        if 'options_data' in st.session_state:
            options_data = st.session_state['options_data']
            
            st.subheader(f"Options Chain for {options_data['symbol']}")
            
            # Metrics display
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Current Price", f"${options_data['current_price']:.2f}")
            with col2_2:
                st.metric("Expiration", options_data['expiration'])
            with col2_3:
                st.metric("Days to Expiry", f"{options_data['days_to_expiry']} days")
            
            # Display calls and puts in tabs
            tab1, tab2 = st.tabs(["Calls", "Puts"])
            
            with tab1:
                if not options_data['calls'].empty:
                    st.dataframe(
                        options_data['calls'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']],
                        use_container_width=True
                    )
            
            with tab2:
                if not options_data['puts'].empty:
                    st.dataframe(
                        options_data['puts'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']],
                        use_container_width=True
                    )
        
        # Options flow analysis display
        if 'flow_data' in st.session_state:
            st.subheader("Options Flow Analysis")
            enhanced_viz.plot_options_flow_dashboard(st.session_state['flow_data'])
    
    # Implied Volatility Surface
    st.header("Implied Volatility Surface")
    st.markdown("""
    <div class="explanation-box">
    <p><strong>Implied Volatility Surface:</strong> This 3D visualization shows how implied volatility varies with strike price (moneyness) and time to expiry. 
    Higher implied volatility indicates higher uncertainty or demand for options at those strike prices and expirations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Generate IV Surface", type="primary"):
        with st.spinner("Calculating implied volatility surface..."):
            iv_surface = options_market_data.calculate_implied_volatility_surface(symbol)
            
            if not iv_surface.empty:
                st.session_state['iv_surface'] = iv_surface
                enhanced_viz.plot_volatility_surface_3d(iv_surface)
                
                # Train ML model on IV data
                if ml_predictor.train_implied_volatility_model(iv_surface):
                    st.success("ML model trained on implied volatility data")
            else:
                st.warning("Could not generate implied volatility surface")
    
    st.stop()
    
elif page == "ML Volatility Forecasting" and ENHANCED_FEATURES_AVAILABLE:
    st.title("ML-Powered Market Analysis")
    
    st.markdown("""
    <div class="explanation-box">
    <h4>AI-Powered Market Analysis</h4>
    <p>This section uses machine learning to forecast market volatility and detect market regimes:</p>
    <ul>
        <li><strong>Volatility Forecasting:</strong> Predicts future market volatility using historical data</li>
        <li><strong>Market Regime Detection:</strong> Identifies current market conditions (Bull, Bear, Sideways, High Volatility)</li>
        <li><strong>Advanced ML Models:</strong> Uses Random Forest and other algorithms for predictions</li>
    </ul>
    <p><strong>How to use:</strong> Enter a stock symbol and forecast horizon, then click "Run ML Analysis".</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("ML Configuration")
        symbol = st.text_input("Symbol for Analysis", value="AAPL")
        forecast_days = st.slider("Forecast Horizon (days)", 1, 60, 30)
        
        if st.button("Run ML Analysis", type="primary", use_container_width=True):
            with st.spinner("Running machine learning analysis..."):
                try:
                    # Fetch historical data using safe method
                    print(f"Fetching data for {symbol}...")
                    hist_data = safe_yfinance_download(symbol, period="2y")
                    
                    if not hist_data.empty:
                        print(f"Successfully fetched {len(hist_data)} data points")
                        
                        # Market regime detection
                        regime = regime_detector.detect_current_regime(hist_data)
                        
                        # Volatility forecasting
                        if vol_forecaster.train_volatility_model(hist_data):
                            forecast_vol = vol_forecaster.forecast_volatility(hist_data, forecast_days)
                            
                            st.session_state['ml_results'] = {
                                'symbol': symbol,
                                'regime': regime,
                                'forecast_vol': forecast_vol,
                                'hist_data': hist_data
                            }
                            
                            st.success("ML analysis complete")
                        else:
                            st.error("Failed to train volatility model")
                    else:
                        st.warning("No historical data available - using fallback data for demonstration")
                        # The safe_yfinance_download function will have provided fallback data
                        regime = "Demonstration Mode"
                        forecast_vol = 0.25  # Default volatility
                        
                        st.session_state['ml_results'] = {
                            'symbol': symbol,
                            'regime': regime,
                            'forecast_vol': forecast_vol,
                            'hist_data': hist_data  # Will be the fallback data
                        }
                        
                        st.info("Analysis complete using demonstration data")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "impersonating" in error_msg.lower() or "chrome" in error_msg.lower():
                        st.error("Data fetching temporarily unavailable in deployment environment.")
                        st.info("This is a known issue with financial data APIs in containers. The system will use demonstration data.")
                        
                        # Provide fallback results
                        st.session_state['ml_results'] = {
                            'symbol': symbol,
                            'regime': "Demo Mode",
                            'forecast_vol': 0.22,
                            'hist_data': None
                        }
                    else:
                        st.error(f"Error in ML analysis: {e}")
    
    with col2:
        if 'ml_results' in st.session_state:
            results = st.session_state['ml_results']
            
            st.subheader(f"AI Analysis for {results['symbol']}")
            
            # Results metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Market Regime", results['regime'])
            with col2_2:
                st.metric("Forecasted Volatility", f"{results['forecast_vol']:.1%}")
            with col2_3:
                current_vol = results['hist_data']['Close'].pct_change().std() * np.sqrt(252)
                vol_change = (results['forecast_vol'] - current_vol) / current_vol
                st.metric("Vol Change", f"{vol_change:+.1%}")
    
    st.stop()

st.title("Black-Scholes Options Pricer")
st.markdown("""
<div class="explanation-box">
<h4>Black-Scholes Options Pricing Model</h4>
<p>This tool calculates theoretical option prices using the Black-Scholes model:</p>
<ul>
    <li><strong>Call/Put Prices:</strong> Theoretical prices for European options</li>
    <li><strong>Heatmaps:</strong> Visualize how option prices change with spot price and volatility</li>
    <li><strong>P&L Analysis:</strong> Profit/Loss visualization based on purchase prices</li>
    <li><strong>Greeks:</strong> Sensitivity measures (Delta, Gamma, Theta, Vega)</li>
</ul>
<p><strong>How to use:</strong> Enter model parameters in the sidebar, then analyze the results in the main panel.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model Inputs")
    symbol = st.text_input("Optional: Stock symbol (e.g. AAPL)", value="")
    live_price = None
    live_vol   = None

    if symbol:
        try:
            tkr  = yf.Ticker(symbol)
            data = tkr.history(period="1mo", interval="1d")["Close"]
            live_price = data.iloc[-1]
            log_rets   = np.log(data / data.shift(1)).dropna()
            live_vol   = log_rets.std() * np.sqrt(252)
            st.success(f"{symbol.upper()}: Spot = {live_price:.2f}, Vol ≈ {live_vol:.2%}")
        except Exception as e:
            st.error(f"Could not fetch data for '{symbol}': {e}")

    default_spot = float(live_price) if live_price is not None else 100.0
    S = st.number_input(
    "Spot price (S)",
    min_value=0.0,
    value=default_spot,
    step=1.0,
    key="spot_input",
    help="Current market price of the underlying asset"
    )

    default_vol = float(live_vol) if live_vol is not None else 0.2
    sigma = st.number_input(
        "Volatility (σ)",
        min_value=0.0,
        value=default_vol,
        step=0.001,
        format="%.3f",
        key="vol_input",
        help="Annualized volatility of the underlying asset (standard deviation of returns)"
    )
    K = st.number_input("Strike price (K)",           min_value=0.0, value=100.0, step=1.0, help="Exercise price of the option")
    T = st.number_input("Time to expiry (T, years)",  min_value=0.0, value=1.0,   step=0.1, help="Time until option expiration (in years)")
    r = st.number_input("Risk-free rate (r)",         min_value=0.0, value=0.05,  step=0.01, help="Risk-free interest rate (e.g., Treasury bill rate)")

    st.divider()
    st.header("Heatmap Ranges")
    st.markdown("Define the range of spot prices and volatilities for heatmap visualization:")
    S_min     = st.number_input("Min spot (Sₘᵢₙ)",    min_value=0.0, value=50.0,  step=1.0, help="Minimum spot price for heatmap")
    S_max     = st.number_input("Max spot (Sₘₐₓ)",    min_value=0.0, value=150.0, step=1.0, help="Maximum spot price for heatmap")
    sigma_min = st.number_input("Min vol (σₘᵢₙ)",     min_value=0.01,value=0.1,   step=0.01, help="Minimum volatility for heatmap")
    sigma_max = st.number_input("Max vol (σₘₐₓ)",     min_value=0.01,value=0.5,   step=0.01, help="Maximum volatility for heatmap")
    st.divider()
    st.header("Grid Resolution")
    N = st.slider(
        "Grid size (N×N points)",
        min_value=50,
        max_value=1000,
        value=300,
        step=25,
        help="Higher values = smoother heatmaps but slower computation"
    )
    
    total_points = N * N
    if total_points < 10000:
        perf_color = "🟢"
        perf_text = "Fast computation (~0.01s)"
    elif total_points < 50000:
        perf_color = "🟡"
        perf_text = "Medium computation (~0.05s)"
    elif total_points < 100000:
        perf_color = "🟠"
        perf_text = "Slower computation (~0.1s)"
    else:
        perf_color = "🔴"
        perf_text = "Slow computation (~0.2s+)"
    
    st.caption(f"{perf_color} **{total_points:,} total calculations** - {perf_text}")
    
    with st.expander("ℹ️ How The Grid Size Affects Performance"):
        st.markdown("""
        **Grid Size Impact:**
        - **N = 100**: 10,000 calculations - Very fast, good for testing
        - **N = 200**: 40,000 calculations - Fast, good balance for most use cases  
        - **N = 300**: 90,000 calculations - Default, smooth heatmaps
        - **N = 400**: 160,000 calculations - High resolution, slower
        - **N = 500**: 250,000 calculations - Maximum detail, slowest
        
        **What happens internally:**
        - Creates an N×N grid of (spot price, volatility) combinations
        - Calculates Black-Scholes price for each combination
        - Higher N = smoother gradients but exponentially more calculations
        - Multi-threading automatically kicks in for N > 224 (50k+ points)
        """)
    
    st.divider()
    st.header("Purchase Prices")
    st.markdown("Enter the prices at which you purchased the options to calculate profit/loss:")
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price  = black_scholes_puts(S, K, T, r, sigma)
    call_buy_price = st.number_input("Call purchase price", min_value=0.0, value=call_price, step=0.01, help="Price at which you bought the call option")
    put_buy_price  = st.number_input("Put purchase price",  min_value=0.0, value=put_price,  step=0.01, help="Price at which you bought the put option")

    imp_vol_call = None
    imp_vol_put  = None

    if call_buy_price > 0:
        try:
            imp_vol_call = implied_volatility(
                call_buy_price, S, K, T, r, is_call=True
            )
        except RuntimeError as e:
            imp_vol_call = f"{e}"

    if put_buy_price > 0:
        try:
            imp_vol_put = implied_volatility(
                put_buy_price, S, K, T, r, is_call=False
            )
        except RuntimeError as e:
            imp_vol_put = f"{e}"

    st.divider()
    st.header("Generate and Download Data")
    st.markdown("Click the button to generate the heatmaps and save their data to a SQLite database. You can then download the database file.")
    save = st.button("Generate & Save to DB")

    if os.path.exists("data.db"):
        with open("data.db", "rb") as f:
            db_bytes = f.read()
        st.download_button(
            label="Download SQLite DB",
            data=db_bytes,
            file_name="data.db",
            mime="application/x-sqlite3"
        )
    else:
        st.warning("No database found. Click ‘Generate & Save’ first.")

S_grid     = np.linspace(S_min,    S_max,    num=N)
sigma_grid = np.linspace(sigma_min, sigma_max, num=N)
S_mat, sigma_mat = np.meshgrid(S_grid, sigma_grid)
call_matrix = black_scholes_optimized(S_mat, K, T, r, sigma_mat, 'call')
put_matrix = black_scholes_optimized(S_mat, K, T, r, sigma_mat, 'put')
call_pnl_matrix = call_matrix - call_buy_price
put_pnl_matrix  = put_matrix  - put_buy_price

max_abs = max(abs(call_pnl_matrix.min()), abs(call_pnl_matrix.max()),abs(put_pnl_matrix.min()),  abs(put_pnl_matrix.max()))

if save:
    create_table()
    params = {
        "stock_price":    S,
        "strike_price":   K,
        "interest_rate":  r,
        "volatility":     sigma,
        "time_to_expiry": T,
        "call_buy_price": call_buy_price,
        "put_buy_price":  put_buy_price
    }
    calc_id = insert_calculation(params)

    sigma_flat = sigma_mat.flatten()
    spot_flat = S_mat.flatten()
    call_pnl_flat = call_pnl_matrix.flatten()
    put_pnl_flat = put_pnl_matrix.flatten()
    n_points = len(sigma_flat)
    sigma_combined = np.concatenate([sigma_flat, sigma_flat])
    spot_combined = np.concatenate([spot_flat, spot_flat])
    pnl_combined = np.concatenate([call_pnl_flat, put_pnl_flat])
    is_call_combined = np.concatenate([np.ones(n_points, dtype=bool), np.zeros(n_points, dtype=bool)])
    bulk_data = list(zip(sigma_combined.astype(float), spot_combined.astype(float), pnl_combined.astype(float), is_call_combined))
    insert_outputs_bulk(calc_id, bulk_data)
    st.success(f"Saved calc_id={calc_id} with {len(bulk_data)} rows using vectorized bulk insert.")

st.header("Options-Price Heatmaps")
st.markdown("""
<div class="explanation-box">
<p><strong>Options Price Heatmaps:</strong> These visualizations show how option prices change across different combinations of spot prices and volatilities. 
Red colors indicate higher option prices, while green colors indicate lower prices.</p>
</div>
""", unsafe_allow_html=True)

# Performance indicator
if NUMBA_AVAILABLE:
    perf_method = "JIT-compiled (Numba)"
    perf_color = "success"
elif N * N >= 50000:
    perf_method = "Multi-threaded (2 cores)"
    perf_color = "info"
else:
    perf_method = "Vectorized (NumPy)"
    perf_color = "warning"

st.info(f"**Performance Mode:** {perf_method}")

col0, col1 = st.columns(2)
with col0:
    st.metric(label="Call Option Price", value=f"{call_price:.4f}")
with col1:
    st.metric(label="Put Option Price",  value=f"{put_price:.4f}")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Price")
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(call_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn_r")
    fig1.colorbar(im1, ax=ax1, label="Price")
    ax1.set_xlabel("Spot Price (S)")
    ax1.set_ylabel("Volatility (σ)")
    safe_pyplot_display(fig1)

with col2:
    st.subheader("Put Option Price")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(put_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn_r")
    fig2.colorbar(im2, ax=ax2, label="Price")
    ax2.set_xlabel("Spot Price (S)")
    ax2.set_ylabel("Volatility (σ)")
    safe_pyplot_display(fig2)

st.header("P&L Heatmaps")
st.markdown("""
<div class="explanation-box">
<p><strong>Profit & Loss Heatmaps:</strong> These visualizations show your potential profit or loss across different combinations of spot prices and volatilities, 
based on the purchase prices you entered. Green areas indicate profit, while red areas indicate loss.</p>
</div>
""", unsafe_allow_html=True)


col0, col1 = st.columns(2)
col2, col3 = st.columns(2)

with col2:
    label = "Call Implied Volatility"
    value = f"{imp_vol_call:.4f}" if isinstance(imp_vol_call, float) else imp_vol_call or "–"
    st.metric(label, value)
with col3:
    label = "Put Implied Volatility"
    value = f"{imp_vol_put:.4f}" if isinstance(imp_vol_put, float) else imp_vol_put or "–"
    st.metric(label, value)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Call P&L")
    fig3, ax3 = plt.subplots()
    im3 = ax3.imshow(call_pnl_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn",
                     vmin=-max_abs, vmax=max_abs)
    fig3.colorbar(im3, ax=ax3, label="P&L")
    ax3.set_xlabel("Spot Price (S)")
    ax3.set_ylabel("Volatility (σ)")
    safe_pyplot_display(fig3)

with col4:
    st.subheader("Put P&L")
    fig4, ax4 = plt.subplots()
    im4 = ax4.imshow(put_pnl_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn",
                     vmin=-max_abs, vmax=max_abs)
    fig4.colorbar(im4, ax=ax4, label="P&L")
    ax4.set_xlabel("Spot Price (S)")
    ax4.set_ylabel("Volatility (σ)")
    safe_pyplot_display(fig4)

st.header("Break-Even Curves")
st.markdown("""
<div class="explanation-box">
<p><strong>Break-Even Curves:</strong> These plots show how your profit/loss changes with the spot price at the midpoint volatility of your heatmaps. 
The dashed horizontal line represents the breakeven point (P&L = 0).</p>
</div>
""", unsafe_allow_html=True)

i_mid     = N // 2
sigma_mid = sigma_grid[i_mid]
S_vals    = S_grid
call_pnl_slice = call_pnl_matrix[i_mid, :]
put_pnl_slice  = put_pnl_matrix[i_mid, :]
colA, colB = st.columns(2)

with colA:
    st.subheader(f"Call Break-Even Curve (σ = {sigma_mid:.3f})")
    fig_call, ax_call = plt.subplots()
    ax_call.plot(S_vals, call_pnl_slice, label="Call P&L", color='#00D4AA', linewidth=2)
    ax_call.axhline(0, color="k", linestyle="--", label="Breakeven")
    ax_call.set_xlabel("Spot price S")
    ax_call.set_ylabel("P&L")
    ax_call.legend()
    ax_call.grid(True, alpha=0.3)
    safe_pyplot_display(fig_call)

with colB:
    st.subheader(f"Put Break-Even Curve (σ = {sigma_mid:.3f})")
    fig_put, ax_put = plt.subplots()
    ax_put.plot(S_vals, put_pnl_slice, label="Put P&L", color='#FF6B6B', linewidth=2)
    ax_put.axhline(0, color="k", linestyle="--", label="Breakeven")
    ax_put.set_xlabel("Spot price S")
    ax_put.set_ylabel("P&L")
    ax_put.legend()
    ax_put.grid(True, alpha=0.3)
    safe_pyplot_display(fig_put)


st.header("Greeks")
st.markdown("""
<div class="explanation-box">
<p><strong>The Greeks:</strong> These measure the sensitivity of option prices to various factors:</p>
<ul>
    <li><strong>Delta:</strong> Sensitivity to changes in the underlying asset price</li>
    <li><strong>Gamma:</strong> Rate of change of Delta</li>
    <li><strong>Theta:</strong> Sensitivity to time decay</li>
    <li><strong>Vega:</strong> Sensitivity to changes in volatility</li>
</ul>
</div>
""", unsafe_allow_html=True)


greek_call = {
    "Delta": delta_call(S, K, T, r, sigma),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega":  vega(S, K, T, r, sigma),
    "Theta": theta_call(S, K, T, r, sigma)
}
greek_put = {
    "Delta": delta_put(S, K, T, r, sigma),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega":  vega(S, K, T, r, sigma),
    "Theta": theta_put(S, K, T, r, sigma)
}

cols_call = st.columns(4)
for col, (name, val) in zip(cols_call, greek_call.items()):
    col.metric(label=f"Call {name}", value=f"{val:.3f}")

cols_put = st.columns(4)
for col, (name, val) in zip(cols_put, greek_put.items()):
    col.metric(label=f"Put {name}", value=f"{val:.3f}")

if ENHANCED_FEATURES_AVAILABLE:
    st.header("Advanced Greeks Analysis")
    st.markdown("""
    <div class="explanation-box">
    <p><strong>Advanced Greeks Analysis:</strong> This visualization shows how the Greeks change across different spot prices, 
    providing insight into how option sensitivities evolve as the underlying asset price moves.</p>
    </div>
    """, unsafe_allow_html=True)
    
    S_range = np.linspace(S_min, S_max, 100)
    greeks_data = {
        'delta_call': [delta_call(s, K, T, r, sigma) for s in S_range],
        'delta_put': [delta_put(s, K, T, r, sigma) for s in S_range],
        'gamma': [gamma(s, K, T, r, sigma) for s in S_range],
        'theta_call': [theta_call(s, K, T, r, sigma) for s in S_range],
        'theta_put': [theta_put(s, K, T, r, sigma) for s in S_range],
        'vega': [vega(s, K, T, r, sigma) for s in S_range]
    }
    
    enhanced_viz.plot_advanced_greeks_dashboard(S_range, greeks_data)


