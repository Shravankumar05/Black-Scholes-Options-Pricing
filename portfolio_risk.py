import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from bs_functions import (black_scholes_call, black_scholes_puts, delta_call, delta_put, gamma, theta_call, theta_put, vega)
from portfolio_utils import (fetch_historical_data, calculate_returns, historical_var, parametric_var, monte_carlo_var, calculate_cvar, stress_test_scenarios, apply_stress_scenario, calculate_portfolio_beta, calculate_maximum_drawdown, calculate_sharpe_ratio, create_correlation_heatmap, analyze_monte_carlo_results)
from portfolio_allocation import portfolio_allocator

def _is_crypto_symbol_simple(symbol):
    crypto_indicators = ['/USDT', '/BUSD', '/BTC', '/ETH', '/BNB']
    return any(indicator in symbol.upper() for indicator in crypto_indicators)

def safe_create_positions(position_data_list):
    positions = []
    for i, pos_data in enumerate(position_data_list):
        try:
            if not all(key in pos_data for key in ['instrument_type', 'symbol', 'quantity']):
                st.error(f"Position {i+1} missing required fields")
                continue
                
            position = PortfolioPosition(**pos_data)
            positions.append(position)
        except Exception as e:
            st.error(f"Error creating position {i+1} ({pos_data.get('symbol', 'Unknown')}): {e}")
            continue
    return positions

class PortfolioPosition:
    OPTIONS_CONTRACT_MULTIPLIER = 100
    
    def __init__(self, instrument_type, symbol, quantity, **kwargs):
        self.instrument_type = instrument_type
        self.symbol = symbol
        self.quantity = quantity
        self.kwargs = kwargs
        
        if 'option' in instrument_type:
            self.strike = float(kwargs.get('strike', 100.0))
            self.time_to_expiry = float(kwargs.get('time_to_expiry', 0.25))
            self.risk_free_rate = float(kwargs.get('risk_free_rate', 0.05))
            self.volatility = float(kwargs.get('volatility', 0.2))
            try:
                self.spot_price = self._get_current_spot_price()
            except Exception as e:
                print(f"Error fetching spot price for {symbol}: {e}")
                self.spot_price = 100.0  # Backup
    
    def _get_current_spot_price(self):
        try:
            if self.instrument_type == 'cryptocurrency' or self._is_crypto_symbol(self.symbol):
                try:
                    from crypto_utils import get_crypto_current_price
                    crypto_price = get_crypto_current_price(self.symbol)
                    if crypto_price:
                        return float(crypto_price)
                except ImportError:
                    st.warning(f"CCXT not installed. Cannot fetch crypto price for {self.symbol}. Using fallback.")
                    crypto_fallback_prices = {
                        'BTC/USDT': 100000.0, 'ETH/USDT': 4400.0, 'BNB/USDT': 310.0,
                        'ADA/USDT': 0.52, 'SOL/USDT': 105.0, 'XRP/USDT': 0.63,
                        'DOT/USDT': 7.2, 'DOGE/USDT': 0.082, 'AVAX/USDT': 37.0,
                        'MATIC/USDT': 0.92
                    }
                    return crypto_fallback_prices.get(self.symbol, 100.0)
                except Exception as e:
                    print(f"Error fetching crypto price for {self.symbol}: {e}")
                    crypto_fallback_prices = {
                        'BTC/USDT': 100000.0, 'ETH/USDT': 4500.0, 'BNB/USDT': 310.0,
                        'ADA/USDT': 0.52, 'SOL/USDT': 105.0, 'XRP/USDT': 0.63,
                        'DOT/USDT': 7.2, 'DOGE/USDT': 0.082, 'AVAX/USDT': 37.0,
                        'MATIC/USDT': 0.92
                    }
                    return crypto_fallback_prices.get(self.symbol, 100.0)
            
            data = yf.download(self.symbol, period="1d", progress=False, auto_adjust=True)
            if not data.empty and 'Close' in data.columns:
                close_price = data['Close'].iloc[-1]
                if hasattr(close_price, 'iloc'):
                    return float(close_price.iloc[0])
                else:
                    return float(close_price)
        except Exception as e:
            print(f"Error fetching price for {self.symbol}: {e}")
        
        if self._is_crypto_symbol(self.symbol):
            return 100.0
        else:
            return 100.0
    
    def _is_crypto_symbol(self, symbol):
        """Check if symbol is cryptocurrency"""
        crypto_indicators = ['/USDT', '/BUSD', '/BTC', '/ETH', '/BNB']
        return any(indicator in symbol.upper() for indicator in crypto_indicators)
    
    def get_current_value(self):
        try:
            if self.instrument_type in ['stock', 'cryptocurrency']:
                current_price = self._get_current_spot_price()
                return self.quantity * current_price
            
            elif self.instrument_type == 'call_option':
                try:
                    option_price = black_scholes_call(float(self.spot_price), float(self.strike), float(self.time_to_expiry), float(self.risk_free_rate), float(self.volatility))
                    return float(self.quantity) * option_price * self.OPTIONS_CONTRACT_MULTIPLIER
                except Exception as e:
                    print(f"Error in Black-Scholes call calculation: {e}")
                    print(f"Parameters: S={self.spot_price}, K={self.strike}, T={self.time_to_expiry}, r={self.risk_free_rate}, σ={self.volatility}")
                    return 0.0
            
            elif self.instrument_type == 'put_option':
                try:
                    option_price = black_scholes_puts(float(self.spot_price), float(self.strike), float(self.time_to_expiry), float(self.risk_free_rate), float(self.volatility))
                    return float(self.quantity) * option_price * self.OPTIONS_CONTRACT_MULTIPLIER
                except Exception as e:
                    print(f"Error in Black-Scholes put calculation: {e}")
                    print(f"Parameters: S={self.spot_price}, K={self.strike}, T={self.time_to_expiry}, r={self.risk_free_rate}, σ={self.volatility}")
                    return 0.0
            
        except Exception as e:
            st.error(f"Error calculating value for {self.symbol}: {e}")
        
        return 0.0
    
    def get_greeks(self):
        greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            if 'option' in self.instrument_type:
                if self.instrument_type == 'call_option':
                    greeks['delta'] = delta_call(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility) * self.quantity * self.OPTIONS_CONTRACT_MULTIPLIER
                    
                    theta_value = theta_call(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility)
                    daily_theta = theta_value / 365.0
                    greeks['theta'] = daily_theta * self.quantity * 100
                    print(f"DEBUG Theta - Raw theta: {theta_value:.6f}, Scaled: {theta_value * self.quantity * 100:.2f}")
                
                elif self.instrument_type == 'put_option':
                    greeks['delta'] = delta_put(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility) * self.quantity * 100
                    
                    theta_value = theta_put(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility)
                    daily_theta = theta_value / 365.0
                    greeks['theta'] = daily_theta * self.quantity * 100
                    print(f"DEBUG Theta - Raw theta: {theta_value:.6f}, Scaled: {theta_value * self.quantity * 100:.2f}")
                
                gamma_value = gamma(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility)
                greeks['gamma'] = gamma_value * self.quantity * 100
                
                vega_value = vega(self.spot_price, self.strike, self.time_to_expiry, self.risk_free_rate, self.volatility)
                greeks['vega'] = vega_value * self.quantity * 100
                
                greeks['rho'] = 0.0
            
            elif self.instrument_type in ['stock', 'cryptocurrency']:
                greeks['delta'] = self.quantity
        
        except Exception as e:
            st.error(f"Error calculating Greeks for {self.symbol}: {e}")
        
        return greeks

def calculate_portfolio_var(positions, confidence_level=0.95, time_horizon=1):
    if not positions:
        return 0.0
    
    try:
        portfolio_value = sum([pos.get_current_value() for pos in positions])
    except Exception as e:
        st.error(f"Error calculating portfolio value: {e}")
        return 0.0
        
    if portfolio_value <= 0:
        return 0.0
    
    symbols = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol') and pos.symbol]))
    if not symbols:
        return 0.0
    
    try:
        price_data = fetch_historical_data(symbols, period="1y")
        if price_data.empty:
            return 0.0
        
        returns = calculate_returns(price_data)
        position_values = {}
        total_stock_value = 0
        
        for pos in positions:
            if pos.instrument_type == 'stock' and hasattr(pos, 'symbol'):
                pos_value = pos.get_current_value()
                position_values[pos.symbol] = position_values.get(pos.symbol, 0) + pos_value
                total_stock_value += pos_value
        
        if total_stock_value > 0:
            portfolio_returns = pd.Series(0.0, index=returns.index)
            for symbol in symbols:
                if symbol in returns.columns and symbol in position_values:
                    weight = position_values[symbol] / total_stock_value
                    portfolio_returns += weight * returns[symbol]
            
            var_return = historical_var(portfolio_returns.values, confidence_level)
            return var_return * portfolio_value
        else:
            if len(symbols) == 1:
                portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else returns
            else:
                portfolio_returns = returns.mean(axis=1)
            
            var_return = historical_var(portfolio_returns.values, confidence_level)
            return var_return * portfolio_value
        
    except Exception as e:
        st.error(f"Error calculating portfolio VaR: {e}")
        return 0.0

def calculate_parametric_portfolio_var(positions, confidence_level=0.95, time_horizon=1):
    if not positions:
        return 0.0
    
    portfolio_value = sum([pos.get_current_value() for pos in positions])
    if portfolio_value <= 0:
        return 0.0
    
    symbols = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol') and pos.symbol]))
    if not symbols:
        return 0.0
    
    try:
        price_data = fetch_historical_data(symbols, period="1y")
        if price_data.empty:
            return 0.0
        
        returns = calculate_returns(price_data)
        position_values = {}
        total_stock_value = 0
        
        for pos in positions:
            if pos.instrument_type in ['stock', 'cryptocurrency'] and hasattr(pos, 'symbol'):
                pos_value = pos.get_current_value()
                position_values[pos.symbol] = position_values.get(pos.symbol, 0) + pos_value
                total_stock_value += pos_value
        
        if total_stock_value > 0:
            portfolio_returns = pd.Series(0.0, index=returns.index)
            for symbol in symbols:
                if symbol in returns.columns and symbol in position_values:
                    weight = position_values[symbol] / total_stock_value
                    portfolio_returns += weight * returns[symbol]
            
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            
            if time_horizon == 1:
                scaled_mean = portfolio_mean
                scaled_std = portfolio_std
            else:
                scaled_mean = portfolio_mean * time_horizon
                scaled_std = portfolio_std * np.sqrt(time_horizon)
            
            var_return = scaled_mean + z_score * scaled_std
            var_dollar = abs(var_return * portfolio_value)
            return var_dollar

        else:
            if len(symbols) == 1:
                portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else returns
            else:
                portfolio_returns = returns.mean(axis=1)
            
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            var_return = portfolio_mean + z_score * portfolio_std
            var_dollar = abs(var_return * portfolio_value)
            return var_dollar
        
    except Exception as e:
        print(f"Error calculating parametric portfolio VaR: {e}")
        return 0.0

def calculate_portfolio_cvar(positions, confidence_level=0.95, time_horizon=1):
    if not positions:
        return 0.0
    
    try:
        portfolio_value = sum([pos.get_current_value() for pos in positions])
    except Exception as e:
        st.error(f"Error calculating portfolio value for CVaR: {e}")
        return 0.0
    if portfolio_value <= 0:
        return 0.0
    
    symbols = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol') and pos.symbol]))
    
    if not symbols:
        return 0.0
    try:
        price_data = fetch_historical_data(symbols, period="1y")
        if price_data.empty:
            return 0.0
        
        returns = calculate_returns(price_data)
        position_values = {}
        total_stock_value = 0
        
        for pos in positions:
            if pos.instrument_type == 'stock' and hasattr(pos, 'symbol'):
                pos_value = pos.get_current_value()
                position_values[pos.symbol] = position_values.get(pos.symbol, 0) + pos_value
                total_stock_value += pos_value
        
        if total_stock_value > 0:
            portfolio_returns = pd.Series(0.0, index=returns.index)
            
            for symbol in symbols:
                if symbol in returns.columns and symbol in position_values:
                    weight = position_values[symbol] / total_stock_value
                    portfolio_returns += weight * returns[symbol]
            
            cvar_return = calculate_cvar(portfolio_returns.values, confidence_level)
            return cvar_return * portfolio_value

        else:
            if len(symbols) == 1:
                portfolio_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else returns
            else:
                portfolio_returns = returns.mean(axis=1)
            
            cvar_return = calculate_cvar(portfolio_returns.values, confidence_level)
            return cvar_return * portfolio_value
        
    except Exception as e:
        st.error(f"Error calculating portfolio CVaR: {e}")
        return 0.0

def calculate_portfolio_greeks(positions):
    portfolio_greeks = {
        'total_delta': 0.0,
        'total_gamma': 0.0,
        'total_theta': 0.0,
        'total_vega': 0.0,
        'total_rho': 0.0
    }
    
    if not positions:
        return portfolio_greeks
    try:
        for position in positions:
            position_greeks = position.get_greeks()
            portfolio_greeks['total_delta'] += position_greeks.get('delta', 0)
            portfolio_greeks['total_gamma'] += position_greeks.get('gamma', 0)
            portfolio_greeks['total_theta'] += position_greeks.get('theta', 0)
            portfolio_greeks['total_vega'] += position_greeks.get('vega', 0)
            portfolio_greeks['total_rho'] += position_greeks.get('rho', 0)
    
    except Exception as e:
        st.error(f"Error calculating portfolio Greeks: {e}")
    
    return portfolio_greeks

def monte_carlo_simulation(positions, num_simulations=10000, time_horizon=1):
    if not positions:
        return np.array([])
    
    try:
        var_value, simulated_portfolio_values, simulated_asset_returns, current_value = monte_carlo_var(positions, num_simulations, time_horizon=time_horizon)
        if current_value > 0:
            return (simulated_portfolio_values - current_value) / current_value
        else:
            return np.random.normal(0, 0.02, num_simulations)
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {e}")
        return np.random.normal(0, 0.02, num_simulations)

def stress_test_portfolio(positions, scenarios):
    if not positions:
        return {}
    
    stress_results = {}
    predefined_scenarios = stress_test_scenarios()
    
    for scenario_name, scenario_params in scenarios.items():
        try:
            if "Market Crash" in scenario_name:
                scenario_key = "2008_financial_crisis"
            elif "Volatility Spike" in scenario_name:
                scenario_key = "covid_crash_2020"
            elif "Interest Rate" in scenario_name:
                scenario_key = "dot_com_bubble"
            else:
                scenario_key = "2008_financial_crisis"
            
            if scenario_key in predefined_scenarios:
                scenario_data = predefined_scenarios[scenario_key]
                stressed_value = apply_stress_scenario(positions, scenario_data)
                stress_results[scenario_name] = stressed_value
            else:
                stress_results[scenario_name] = 0.0
                
        except Exception as e:
            st.error(f"Error in stress test {scenario_name}: {e}")
            stress_results[scenario_name] = 0.0
    
    return stress_results

def calculate_correlation_matrix(symbols, period="1y"):
    n = len(symbols)
    return np.eye(n)

def render_portfolio_risk_page():
    st.title("Portfolio Risk Analysis")
    st.markdown("Analyze risk metrics for your multi-instrument portfolio")
    
    try:
        import ccxt
        exchange = ccxt.binance()
        crypto_status = f"Crypto support enabled (CCXT v{ccxt.__version__})"
        crypto_color = "success"
        with st.expander("🔧 Crypto Connection Test"):
            if st.button("Test Binance Connection"):
                try:
                    markets = exchange.load_markets()
                    st.success(f"Connected to Binance! Found {len(markets)} markets")
                    st.write("Sample markets:", list(markets.keys())[:10])
                except Exception as e:
                    st.error(f"Connection test failed: {e}")
                    
    except ImportError as e:
        crypto_status = f"CCXT not found. Install with: `pip install ccxt` (Error: {e})"
        crypto_color = "warning"
    except Exception as e:
        crypto_status = f"CCXT error: {e}"
        crypto_color = "warning"
    
    st.info(f"{crypto_status}")
    
    if 'portfolio_positions' not in st.session_state:
        st.session_state.portfolio_positions = []
    
    with st.sidebar:
        st.header("Portfolio Construction")
        with st.expander("➕ Add New Position", expanded=True):
            instrument_type = st.selectbox(
                "Instrument Type",
                ["Stock", "Cryptocurrency", "Call Option", "Put Option"],
                key="new_instrument_type"
            )
            
            if instrument_type == "Cryptocurrency":
                symbol = st.selectbox(
                    "Cryptocurrency Symbol",
                    options=["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", 
                            "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT",
                            "Custom..."],
                    key="crypto_symbol"
                )
                
                if symbol == "Custom...":
                    symbol = st.text_input("Enter crypto symbol (e.g., LINK/USDT)", key="custom_crypto")
            else:
                symbol = st.text_input("Symbol (e.g., AAPL for stocks)", key="new_symbol")
            quantity = st.number_input("Quantity", value=100, step=1, key="new_quantity")
            
            if "Option" in instrument_type and instrument_type != "Cryptocurrency":
                col1, col2 = st.columns(2)
                with col1:
                    strike = st.number_input("Strike Price", value=100.0, step=1.0, key="new_strike")
                with col2:
                    days_to_expiry = st.number_input("Days to Expiry", value=30, step=1, key="new_days")
                
                expiry_date = datetime.now() + timedelta(days=days_to_expiry)
                time_to_expiry = days_to_expiry / 365.0
                
                col3, col4 = st.columns(2)
                with col3:
                    risk_free_rate = st.number_input("Risk-free Rate", value=0.05, step=0.01, key="new_rate")
                with col4:
                    auto_vol = 0.2  # Default
                    if symbol and symbol.strip():
                        try:
                            ticker_data = yf.download(symbol.strip().upper(), period="1y", progress=False, auto_adjust=True)
                            if not ticker_data.empty and 'Close' in ticker_data.columns:
                                close_prices = ticker_data['Close']
                                if len(close_prices) > 1:
                                    returns = np.log(close_prices / close_prices.shift(1)).dropna()
                                    if len(returns) > 0:
                                        vol_value = returns.std() * np.sqrt(252)  # Annualized volatility
                                        
                                        if hasattr(vol_value, 'iloc'):
                                            auto_vol = float(vol_value.iloc[0])
                                        elif hasattr(vol_value, 'item'):
                                            auto_vol = float(vol_value.item())
                                        else:
                                            auto_vol = float(vol_value)
                                        
                                        if 0.05 <= auto_vol <= 2.0:
                                            st.info(f"📊 Auto-fetched volatility for {symbol}: {auto_vol:.1%}")
                                        else:
                                            auto_vol = 0.2  # Reset to default if unrealistic
                        except Exception as e:
                            print(f"Error fetching volatility for {symbol}: {e}")
                            auto_vol = 0.2  # Ensure we have a fallback
                    
                    volatility = st.number_input(
                        "Volatility", 
                        value=float(auto_vol), 
                        step=0.01, 
                        key="new_vol",
                        help="Automatically fetched from historical data, but you can adjust it"
                    )
            
            if st.button("Add Position"):
                if symbol:
                    # Handle crypto symbols differently
                    if instrument_type == "Cryptocurrency":
                        position_data = {
                            'instrument_type': 'cryptocurrency',
                            'symbol': symbol.upper(),
                            'quantity': quantity
                        }
                    else:
                        position_data = {
                            'instrument_type': instrument_type.lower().replace(" ", "_"),
                            'symbol': symbol.upper(),
                            'quantity': quantity
                        }
                        
                        if "Option" in instrument_type:
                            position_data.update({
                                'strike': strike,
                                'time_to_expiry': time_to_expiry,
                                'risk_free_rate': risk_free_rate,
                                'volatility': volatility
                            })
                    
                    st.session_state.portfolio_positions.append(position_data)
                    st.success(f"Added {quantity} {symbol} {instrument_type}")
                    st.rerun()
        
        st.divider()
        
        # Risk Analysis Parameters
        st.header("Risk Parameters")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        time_horizon = st.selectbox("Time Horizon (days)", [1, 5, 10, 22], index=0)
        num_simulations = st.selectbox("Monte Carlo Simulations", [1000, 5000, 10000], index=2)
        
        st.divider()
        
        # Enhanced Stress Test Scenarios
        st.header("🔥 Stress Test Scenarios")
        
        # Predefined scenarios with descriptions
        st.subheader("Historical Scenarios")
        historical_scenarios = st.multiselect(
            "Select Historical Crisis Scenarios",
            [
                "2008 Financial Crisis (-37% stocks, +150% vol)",
                "COVID-19 Crash 2020 (-34% stocks, +200% vol)", 
                "Dot-com Bubble 2000 (-49% tech, +130% vol)",
                "Black Monday 1987 (-22% in one day)"
            ],
            default=["2008 Financial Crisis (-37% stocks, +150% vol)"],
            help="Based on actual historical market crashes"
        )
        
        st.subheader("Hypothetical Scenarios")
        hypothetical_scenarios = st.multiselect(
            "Select Hypothetical Stress Scenarios",
            [
                "Market Crash (-20% equities)",
                "Volatility Spike (+50% volatility)",
                "Interest Rate Shock (+200 bps)",
                "Crypto Winter (-80% crypto)",
                "Currency Crisis (+20% USD strength)",
                "Inflation Shock (+5% inflation)"
            ],
            help="Theoretical scenarios for forward-looking stress testing"
        )
        
        # Custom scenario builder
        with st.expander("🛠️ Build Custom Scenario"):
            st.write("**Create Your Own Stress Test**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                custom_equity_shock = st.slider(
                    "Equity Shock (%)", 
                    min_value=-80, max_value=50, value=-25, step=5,
                    help="Percentage change in stock prices"
                )
            
            with col2:
                custom_vol_multiplier = st.slider(
                    "Volatility Multiplier", 
                    min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                    help="Multiply current volatility by this factor"
                )
            
            with col3:
                custom_rate_change = st.slider(
                    "Interest Rate Change (bps)", 
                    min_value=-500, max_value=500, value=200, step=25,
                    help="Basis points change in interest rates"
                )
            
            if st.button("Add Custom Scenario"):
                custom_name = f"Custom: {custom_equity_shock:+}% equity, {custom_vol_multiplier:.1f}x vol, {custom_rate_change:+}bps rates"
                if 'custom_scenarios' not in st.session_state:
                    st.session_state.custom_scenarios = []
                st.session_state.custom_scenarios.append({
                    'name': custom_name,
                    'equity_shock': custom_equity_shock / 100,
                    'vol_multiplier': custom_vol_multiplier,
                    'rate_change': custom_rate_change / 10000
                })
                st.success(f"Added: {custom_name}")
        
        # Combine all selected scenarios
        all_scenarios = historical_scenarios + hypothetical_scenarios
        if hasattr(st.session_state, 'custom_scenarios'):
            all_scenarios.extend([s['name'] for s in st.session_state.custom_scenarios])
    
    # Main content area
    if not st.session_state.portfolio_positions:
        st.info("👈 Add positions to your portfolio using the sidebar to begin risk analysis")
        return
    
    # Display current portfolio
    st.header("Current Portfolio")
    
    if st.session_state.portfolio_positions:
        portfolio_df = pd.DataFrame(st.session_state.portfolio_positions)
        
        # Add action column for removing positions
        col1, col2 = st.columns([4, 1])
        with col1:
            st.dataframe(portfolio_df, use_container_width=True)
        
        with col2:
            st.write("Actions")
            for i, position in enumerate(st.session_state.portfolio_positions):
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.portfolio_positions.pop(i)
                    st.rerun()
        
        if st.button("Clear All Positions"):
            st.session_state.portfolio_positions = []
            st.rerun()
    else:
        st.info("No positions in portfolio yet.")
    
    st.divider()
    
    # Portfolio Summary
    st.header("Portfolio Summary")
    
    # Calculate portfolio positions and total value
    # Convert dictionaries to PortfolioPosition objects safely
    positions = safe_create_positions(st.session_state.portfolio_positions)
    
    if not positions:
        st.info("👈 Add positions to your portfolio using the sidebar to begin risk analysis")
        return
    
    # Calculate total portfolio value
    total_portfolio_value = sum([pos.get_current_value() for pos in positions])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}", border=True)
    with col2:
        st.metric("Number of Positions", len(positions), border=True)
    with col3:
        unique_symbols = len(set([pos.symbol for pos in positions]))
        st.metric("Unique Instruments", unique_symbols, border=True)
    
    st.divider()
    
    # Risk Metrics Dashboard
    st.header("Risk Metrics Dashboard")
    
    # Portfolio-level risk metrics
    st.subheader("Portfolio Risk Metrics")
    
    # Calculate portfolio value and basic stats for parametric VaR
    portfolio_value = sum([pos.get_current_value() for pos in positions])
    
    # VaR Methods Comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_value = calculate_portfolio_var(positions, confidence_level, time_horizon)
        st.metric("Historical VaR", f"${var_value:,.2f}", 
                 help=f"VaR based on historical return distribution at {confidence_level:.0%} confidence", border=True)
    
    with col2:
        # Calculate parametric VaR using proper portfolio approach
        parametric_var_value = calculate_parametric_portfolio_var(positions, confidence_level, time_horizon)
        st.metric("Parametric VaR", f"${parametric_var_value:,.2f}", 
                 help=f"VaR assuming normal distribution of returns at {confidence_level:.0%} confidence", border=True)
    
    with col3:
        cvar_value = calculate_portfolio_cvar(positions, confidence_level, time_horizon)
        st.metric("Conditional VaR (CVaR)", f"${cvar_value:,.2f}", 
                 help=f"Expected loss given that loss exceeds VaR (tail risk)", border=True)
    
    # VaR Methods Comparison
    st.subheader("📊 VaR Methods Comparison")
    
    # Create comparison table
    var_comparison_data = {
        "Method": ["Historical VaR", "Parametric VaR", "Monte Carlo VaR (95%)"],
        "Value": [f"${var_value:,.0f}", f"${parametric_var_value:,.0f}", "Run simulation →"],
        "Approach": [
            "Uses actual historical returns distribution",
            "Assumes normal distribution of returns", 
            "Simulates thousands of future scenarios"
        ],
        "Pros": [
            "No distribution assumptions, captures actual tail events",
            "Fast calculation, smooth estimates",
            "Handles complex portfolios, any distribution"
        ],
        "Cons": [
            "Limited by historical data, assumes future = past",
            "May underestimate tail risk, normal assumption",
            "Computationally intensive, model dependent"
        ]
    }
    
    comparison_df = pd.DataFrame(var_comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Explain differences if they're large
    if var_value > 0 and parametric_var_value > 0:
        ratio = max(var_value, parametric_var_value) / min(var_value, parametric_var_value)
        if ratio > 2.0:
            st.warning(f"⚠️ **Large VaR Difference Detected** (ratio: {ratio:.1f}x)")
            st.write("**Possible reasons:**")
            if parametric_var_value > var_value:
                st.write("• **Parametric VaR > Historical VaR**: Returns may have fat tails or be skewed")
                st.write("• **Normal distribution assumption** may be inappropriate for this portfolio")
            else:
                st.write("• **Historical VaR > Parametric VaR**: Historical data includes extreme events")
                st.write("• **Recent volatility** may be higher than long-term average")
            st.write("• **Portfolio composition**: Options and crypto have non-normal return distributions")
    
    # Portfolio Greeks
    st.subheader("Portfolio Greeks")
    col4, col5, col6, col7 = st.columns(4)
    
    portfolio_greeks = calculate_portfolio_greeks(positions)
    
    with col4:
        st.metric("Portfolio Delta", f"{portfolio_greeks['total_delta']:.0f}", 
                 help="Equivalent shares exposure", border=True)
    
    with col5:
        st.metric("Portfolio Gamma", f"{portfolio_greeks['total_gamma']:.0f}", 
                 help="Delta change per $1 stock move", border=True)
    
    with col6:
        st.metric("Portfolio Theta", f"{portfolio_greeks['total_theta']:.0f}", 
                 help="Daily time decay in dollars", border=True)
    
    with col7:
        st.metric("Portfolio Vega", f"{portfolio_greeks['total_vega']:.0f}", 
                 help="P&L change per 1% volatility move", border=True)
    

    
    # Individual position risk metrics
    st.subheader("Individual Position Risk")
    
    if len(positions) > 0:
        # Create a DataFrame for individual position metrics
        position_data = []
        
        for i, position in enumerate(positions):
            try:
                # Calculate individual position VaR (simplified approach)
                position_value = position.get_current_value()
                
                # For individual VaR, we'll use a simplified approach based on position volatility
                if hasattr(position, 'symbol') and position.symbol:
                    try:
                        # Fetch individual stock data
                        price_data = fetch_historical_data([position.symbol], period="1y")
                        if not price_data.empty:
                            returns = calculate_returns(price_data)
                            if len(returns) > 0:
                                individual_var = historical_var(returns.iloc[:, 0].values, confidence_level)
                                individual_var_dollar = individual_var * position_value
                            else:
                                individual_var_dollar = 0.0
                        else:
                            individual_var_dollar = 0.0
                    except:
                        individual_var_dollar = 0.0
                else:
                    individual_var_dollar = 0.0
                
                # Get position Greeks
                position_greeks = position.get_greeks()
                
                position_data.append({
                    'Position': f"{position.quantity} {position.symbol} {position.instrument_type.replace('_', ' ').title()}",
                    'Current Value': f"${position_value:,.2f}",
                    'Individual VaR': f"${individual_var_dollar:,.2f}",
                    'Delta': f"{position_greeks.get('delta', 0):.2f}",
                    'Gamma': f"{position_greeks.get('gamma', 0):.2f}",
                    'Theta': f"{position_greeks.get('theta', 0):.2f}",
                    'Vega': f"{position_greeks.get('vega', 0):.2f}"
                })
            except Exception as e:
                st.error(f"Error calculating metrics for position {i+1}: {e}")
        
        if position_data:
            position_df = pd.DataFrame(position_data)
            st.dataframe(position_df, use_container_width=True)
        else:
            st.info("Unable to calculate individual position metrics")
    else:
        st.info("No positions to analyze")
    
    st.divider()
    
    # Enhanced Correlation Analysis
    st.header("📊 Asset Correlation Analysis")
    
    symbols = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol') and pos.symbol]))
    
    # Separate crypto and stock symbols for better analysis
    crypto_symbols = [s for s in symbols if _is_crypto_symbol_simple(s)]
    stock_symbols = [s for s in symbols if not _is_crypto_symbol_simple(s)]
    
    if len(symbols) >= 2:
        # Show portfolio composition
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assets", len(symbols), border=True)
        with col2:
            st.metric("Traditional Assets", len(stock_symbols), border=True)
        with col3:
            st.metric("Crypto Assets", len(crypto_symbols), border=True)
        
        # Correlation analysis with better error handling
        correlation_matrix, symbol_list = create_correlation_heatmap(symbols, period="1y")
        
        if correlation_matrix is not None and not correlation_matrix.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create correlation heatmap with matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create the heatmap
                im = ax.imshow(correlation_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Set ticks and labels
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix.index)))
                ax.set_xticklabels(correlation_matrix.columns)
                ax.set_yticklabels(correlation_matrix.index)
                
                # Add correlation values as text
                for i in range(len(correlation_matrix.index)):
                    for j in range(len(correlation_matrix.columns)):
                        value = correlation_matrix.iloc[i, j]
                        color = 'white' if abs(value) > 0.5 else 'black'
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=color, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Correlation Coefficient')
                
                ax.set_title('Asset Correlation Matrix')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("**Correlation Insights:**")
                
                # Find highest and lowest correlations
                corr_values = correlation_matrix.values
                np.fill_diagonal(corr_values, np.nan)  # Remove self-correlations
                
                if not np.all(np.isnan(corr_values)):
                    max_corr = np.nanmax(corr_values)
                    min_corr = np.nanmin(corr_values)
                    avg_corr = np.nanmean(corr_values)
                    
                    st.metric("Highest Correlation", f"{max_corr:.3f}", border=True)
                    st.metric("Lowest Correlation", f"{min_corr:.3f}", border=True)
                    st.metric("Average Correlation", f"{avg_corr:.3f}", border=True)
                    
                    # Interpretation
                    if avg_corr > 0.7:
                        st.warning("⚠️ High correlation - Limited diversification benefit")
                    elif avg_corr > 0.3:
                        st.info("ℹ️ Moderate correlation - Some diversification benefit")
                    else:
                        st.success("✅ Low correlation - Good diversification")
        else:
            st.warning("Unable to fetch correlation data")
            
            if crypto_symbols and not CRYPTO_AVAILABLE:
                st.error("**Crypto Support Missing**: Install CCXT to analyze crypto correlations")
                st.code("pip install ccxt", language="bash")
            
            if crypto_symbols and stock_symbols:
                st.info("**Mixed Portfolio Detected**: Analyzing both traditional and crypto assets")
                st.write("**Possible Issues:**")
                st.write("• Crypto symbols need CCXT library")
                st.write("• Different data sources (Yahoo Finance vs Binance)")
                st.write("• Limited historical overlap between asset classes")
            
            st.write("**Troubleshooting:**")
            st.write("• Check symbol formats (stocks: AAPL, crypto: BTC/USDT)")
            st.write("• Verify internet connection")
            st.write("• Try with fewer symbols")
            st.write("• Install missing dependencies")
    else:
        st.info("Add at least 2 different assets to see correlation analysis.")
    
    st.divider()
    
    st.header("Monte Carlo Simulation")
    
    if st.button("Run Monte Carlo Simulation"):
        if num_simulations >= 1000:
            try:
                from numba import jit
                perf_info = f"🚀 JIT-accelerated simulation ({num_simulations:,} simulations)"
                perf_color = "success"
            except ImportError:
                perf_info = f"🔢 Standard simulation ({num_simulations:,} simulations)"
                perf_color = "info"
        else:
            perf_info = f"🔢 Standard simulation ({num_simulations:,} simulations)"
            perf_color = "info"
        
        st.info(perf_info)
        
        with st.spinner("Running Monte Carlo simulation..."):
            # Run enhanced Monte Carlo simulation
            var_dollar, simulated_portfolio_values, simulated_asset_returns, current_value = monte_carlo_var(
                positions, num_simulations, confidence_level, time_horizon
            )
            
            # Analyze results
            mc_results = analyze_monte_carlo_results(simulated_portfolio_values, current_value, time_horizon)
            
            # Display key statistics
            st.subheader("Simulation Results Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Portfolio Value", f"${current_value:,.0f}", border=True)
            with col2:
                st.metric("Mean Simulated Value", f"${mc_results['mean_simulated_value']:,.0f}", 
                         f"{((mc_results['mean_simulated_value']/current_value-1)*100):+.1f}%", border=True)
            with col3:
                st.metric("Best Case (95th %ile)", f"${np.percentile(simulated_portfolio_values, 95):,.0f}", border=True)
            with col4:
                st.metric("Worst Case (5th %ile)", f"${np.percentile(simulated_portfolio_values, 5):,.0f}", border=True)
            
            # VaR metrics for different confidence levels
            st.subheader("Value at Risk Analysis")
            var_col1, var_col2, var_col3 = st.columns(3)
            
            with var_col1:
                st.metric("Monte Carlo VaR (90%)", f"${mc_results['var_90']:,.0f}", 
                         help="Simulation-based VaR at 90% confidence", border=True)
            with var_col2:
                st.metric("Monte Carlo VaR (95%)", f"${mc_results['var_95']:,.0f}", 
                         help="Simulation-based VaR at 95% confidence", border=True)
            with var_col3:
                st.metric("Monte Carlo VaR (99%)", f"${mc_results['var_99']:,.0f}", 
                         help="Simulation-based VaR at 99% confidence", border=True)
            
            # Create visualizations
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Portfolio Value Distribution
            portfolio_returns = (simulated_portfolio_values - current_value) / current_value * 100
            ax1.hist(portfolio_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            
            # Add VaR lines
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            ax1.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.1f}%')
            ax1.axvline(var_99, color='darkred', linestyle='--', linewidth=2, label=f'99% VaR: {var_99:.1f}%')
            ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
            
            ax1.set_xlabel('Portfolio Return (%)')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Distribution of Portfolio Returns')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Portfolio Value vs Current Value
            ax2.scatter(range(min(1000, len(simulated_portfolio_values))), 
                       simulated_portfolio_values[:1000], alpha=0.6, s=1)
            ax2.axhline(current_value, color='red', linestyle='--', label=f'Current Value: ${current_value:,.0f}')
            ax2.axhline(np.percentile(simulated_portfolio_values, 5), color='orange', 
                       linestyle='--', label=f'5th Percentile: ${np.percentile(simulated_portfolio_values, 5):,.0f}')
            ax2.set_xlabel('Simulation Number')
            ax2.set_ylabel('Portfolio Value ($)')
            ax2.set_title('Simulated Portfolio Values (First 1000)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Asset Returns Correlation (if multiple assets)
            if simulated_asset_returns.shape[1] > 1:
                symbols = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol')]))[:2]  # First 2 symbols
                if len(symbols) >= 2:
                    returns_1 = simulated_asset_returns[:, 0] * 100
                    returns_2 = simulated_asset_returns[:, 1] * 100
                    ax3.scatter(returns_1, returns_2, alpha=0.5, s=1)
                    ax3.set_xlabel(f'{symbols[0]} Return (%)')
                    ax3.set_ylabel(f'{symbols[1]} Return (%)')
                    ax3.set_title('Simulated Asset Returns Correlation')
                    ax3.grid(True, alpha=0.3)
                    
                    # Add correlation coefficient
                    corr_coef = np.corrcoef(returns_1, returns_2)[0, 1]
                    ax3.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                            transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                else:
                    ax3.text(0.5, 0.5, 'Need multiple assets\nfor correlation plot', 
                            ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'Single asset portfolio\nNo correlation to show', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # 4. Risk Metrics Comparison
            risk_metrics = ['MC VaR 90%', 'MC VaR 95%', 'MC VaR 99%', 'MC CVaR 95%']
            risk_values = [mc_results['var_90'], mc_results['var_95'], 
                          mc_results['var_99'], mc_results['cvar_95']]
            
            bars = ax4.bar(risk_metrics, risk_values, color=['lightblue', 'orange', 'red', 'darkred'])
            ax4.set_ylabel('Loss Amount ($)')
            ax4.set_title('Risk Metrics Comparison')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, risk_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'${value:,.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistical Analysis
            st.subheader("Statistical Analysis")
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write("**Distribution Statistics:**")
                if time_horizon == 1:
                    st.write(f"• Daily Mean Return: {mc_results['mean_return']*100:.2f}%")
                    st.write(f"• Daily Volatility: {mc_results['volatility']*100:.2f}%")
                    st.write(f"• Annualized Return: {mc_results['annual_mean_return']*100:.1f}%")
                    st.write(f"• Annualized Volatility: {mc_results['annual_volatility']*100:.1f}%")
                else:
                    st.write(f"• {time_horizon}-day Mean Return: {mc_results['mean_return']*100:.2f}%")
                    st.write(f"• {time_horizon}-day Volatility: {mc_results['volatility']*100:.2f}%")
                st.write(f"• Skewness: {mc_results['skewness']:.3f}")
                st.write(f"• Kurtosis: {mc_results['kurtosis']:.3f}")
            
            with stats_col2:
                st.write("**Risk Analysis:**")
                prob_loss = (simulated_portfolio_values < current_value).mean() * 100
                prob_large_loss = (simulated_portfolio_values < current_value * 0.9).mean() * 100
                st.write(f"• Probability of Loss: {prob_loss:.1f}%")
                st.write(f"• Probability of >10% Loss: {prob_large_loss:.1f}%")
                st.write(f"• Maximum Simulated Loss: ${current_value - mc_results['min_value']:,.0f}")
                st.write(f"• Maximum Simulated Gain: ${mc_results['max_value'] - current_value:,.0f}")
    
    st.divider()
    
    # Stress Testing Results
    st.header("Stress Testing Results")
    
    if all_scenarios:
        if st.button("🚀 Run Stress Tests", type="primary"):
            with st.spinner("Running comprehensive stress tests..."):
                # Map scenarios to parameters
                stress_scenarios = {}
                
                for scenario in all_scenarios:
                    if "2008 Financial Crisis" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.37, 'vol_increase': 2.5, 'rate_change': -0.03}
                    elif "COVID-19 Crash" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.34, 'vol_increase': 3.0, 'rate_change': -0.015}
                    elif "Dot-com Bubble" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.49, 'vol_increase': 2.3, 'rate_change': -0.025}
                    elif "Black Monday" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.22, 'vol_increase': 4.0, 'rate_change': 0.0}
                    elif "Market Crash (-20%)" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.20, 'vol_increase': 1.5, 'rate_change': 0.0}
                    elif "Volatility Spike" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': 0.0, 'vol_increase': 1.5, 'rate_change': 0.0}
                    elif "Interest Rate Shock" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.05, 'vol_increase': 1.2, 'rate_change': 0.02}
                    elif "Crypto Winter" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.10, 'vol_increase': 2.0, 'rate_change': 0.0}
                    elif "Currency Crisis" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.15, 'vol_increase': 1.8, 'rate_change': 0.01}
                    elif "Inflation Shock" in scenario:
                        stress_scenarios[scenario] = {'equity_shock': -0.12, 'vol_increase': 1.3, 'rate_change': 0.03}
                
                # Add custom scenarios
                if hasattr(st.session_state, 'custom_scenarios'):
                    for custom in st.session_state.custom_scenarios:
                        if custom['name'] in all_scenarios:
                            stress_scenarios[custom['name']] = {
                                'equity_shock': custom['equity_shock'],
                                'vol_increase': custom['vol_multiplier'],
                                'rate_change': custom['rate_change']
                            }
                
                # Run stress tests
                stress_results = stress_test_portfolio(positions, stress_scenarios)
                
                # Enhanced results display
                if stress_results:
                    st.subheader("📈 Stress Test Results")
                    
                    # Create results DataFrame with more details
                    results_data = []
                    current_value = sum([pos.get_current_value() for pos in positions])
                    
                    for scenario, impact in stress_results.items():
                        impact_pct = (impact / current_value * 100) if current_value > 0 else 0
                        results_data.append({
                            "Scenario": scenario,
                            "Portfolio Impact ($)": f"${impact:,.0f}",
                            "Impact (%)": f"{impact_pct:+.1f}%",
                            "Severity": "🔴 Severe" if abs(impact_pct) > 20 else "🟡 Moderate" if abs(impact_pct) > 10 else "🟢 Mild"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    impacts = list(stress_results.values())
                    if impacts:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Worst Case Loss", f"${min(impacts):,.0f}", border=True)
                        with col2:
                            st.metric("Average Impact", f"${np.mean(impacts):,.0f}", border=True)
                        with col3:
                            st.metric("Best Case", f"${max(impacts):,.0f}", border=True)
                else:
                    st.error("Unable to run stress tests. Check your positions and try again.")
    else:
        st.info("👆 Select stress test scenarios above to analyze portfolio resilience")
    
    st.divider()
    
    st.header("Portfolio Allocation Recommendations")
    portfolio_assets = list(set([pos.symbol for pos in positions if hasattr(pos, 'symbol')]))
    
    if len(portfolio_assets) >= 2:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("Allocation Settings")
            
            risk_profile = st.selectbox(
                "Risk Profile",
                ["Conservative", "Moderate", "Aggressive"],
                index=1,
                help="Risk tolerance determines target return and volatility constraints"
            )
            
            market_regime = st.selectbox(
                "Market Regime",
                ["Normal", "Bull Market", "Bear Market", "High Volatility"],
                index=0,
                help="Current market conditions affect allocation strategy"
            )
            
            current_allocations = {}
            total_value = sum([pos.get_current_value() for pos in positions])
            for pos in positions:
                if hasattr(pos, 'symbol') and total_value > 0:
                    weight = pos.get_current_value() / total_value
                    current_allocations[pos.symbol] = current_allocations.get(pos.symbol, 0) + weight
            
            if st.button("Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing portfolio allocation strategies..."):
                    try:
                        price_data = fetch_historical_data(portfolio_assets, period="1y")
                        if not price_data.empty:
                            returns_data = calculate_returns(price_data)
                            
                            recommendations = portfolio_allocator.get_allocation_recommendation(returns_data=returns_data, assets=portfolio_assets,
                                risk_profile=risk_profile.lower(),
                                current_allocations=current_allocations,
                                market_regime=market_regime.lower().replace(' ', '_')
                            )
                            
                            st.session_state['allocation_recommendations'] = recommendations
                            st.success("Allocation analysis complete!")
                        else:
                            st.error("Unable to fetch sufficient historical data for analysis")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if 'allocation_recommendations' in st.session_state:
                recommendations = st.session_state['allocation_recommendations']
                
                st.subheader("Allocation Strategy Rankings")
                
                for i, (strategy_name, strategy_data) in enumerate(recommendations['ranked_strategies'][:3]):
                    with st.expander(f"#{i+1}: {strategy_name} (Score: {strategy_data['suitability_score']:.1f})", expanded=(i==0)):
                        
                        col2_1, col2_2, col2_3 = st.columns(3)
                        with col2_1:
                            st.metric("Expected Return", f"{strategy_data['metrics']['return']:.1%}")
                        with col2_2:
                            st.metric("Volatility", f"{strategy_data['metrics']['volatility']:.1%}")
                        with col2_3:
                            st.metric("Sharpe Ratio", f"{strategy_data['metrics']['sharpe_ratio']:.2f}")
                        
                        st.write("**Recommended Allocation:**")
                        allocation_df = pd.DataFrame([
                            {'Asset': asset, 'Weight': f"{weight:.1%}", 'Dollar Amount': f"${weight * total_value:,.0f}"}
                            for asset, weight in strategy_data['allocation'].items()
                            if weight > 0.01  # Only show allocations > 1%
                        ])
                        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                
                st.subheader("Rebalancing Suggestions")
                if recommendations['top_recommendation']:
                    top_strategy = recommendations['top_recommendation'][1]['allocation']
                    rebalancing_suggestions = portfolio_allocator.rebalancing_suggestions(
                        current_allocations, top_strategy, threshold=0.05
                    )
                    
                    if rebalancing_suggestions:
                        st.write(f"**Based on {recommendations['top_recommendation'][0]} strategy:**")
                        
                        rebalance_df = pd.DataFrame(rebalancing_suggestions)
                        rebalance_df = rebalance_df[['asset', 'action', 'current_weight', 'target_weight', 'difference']]
                        rebalance_df['current_weight'] = rebalance_df['current_weight'].apply(lambda x: f"{x:.1%}")
                        rebalance_df['target_weight'] = rebalance_df['target_weight'].apply(lambda x: f"{x:.1%}")
                        rebalance_df['difference'] = rebalance_df['difference'].apply(lambda x: f"{x:+.1%}")
                        rebalance_df.columns = ['Asset', 'Action', 'Current %', 'Target %', 'Change']
                        
                        st.dataframe(rebalance_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Portfolio is well-balanced. No major rebalancing needed.")
                
                st.subheader("Strategy Comparison")
                strategy_comparison = []
                for strategy_name, strategy_data in recommendations['recommendations'].items():
                    strategy_comparison.append({
                        'Strategy': strategy_name,
                        'Return': strategy_data['metrics']['return'],
                        'Volatility': strategy_data['metrics']['volatility'],
                        'Sharpe Ratio': strategy_data['metrics']['sharpe_ratio'],
                        'Suitability Score': strategy_data['suitability_score']
                    })
                
                comparison_df = pd.DataFrame(strategy_comparison)
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(comparison_df['Volatility'], comparison_df['Return'], 
                                   s=comparison_df['Suitability Score']*10, 
                                   c=comparison_df['Suitability Score'], 
                                   cmap='viridis', alpha=0.7)
                
                for i, strategy in enumerate(comparison_df['Strategy']):
                    ax.annotate(strategy, (comparison_df['Volatility'][i], comparison_df['Return'][i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.set_xlabel('Volatility')
                ax.set_ylabel('Expected Return')
                ax.set_title('Risk-Return Profile of Allocation Strategies')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, label='Suitability Score')
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("Risk Profile Insights"):
                    profile_data = portfolio_allocator.risk_tolerance_profiles.get(risk_profile.lower())
                    if profile_data:
                        st.write(f"**{risk_profile} Risk Profile:**")
                        st.write(f"• Target Annual Return: {profile_data['target_return']:.1%}")
                        st.write(f"• Maximum Volatility: {profile_data['max_volatility']:.1%}")
                        
                        current_metrics = recommendations['top_recommendation'][1]['metrics']
                        if current_metrics['volatility'] > profile_data['max_volatility']:
                            st.warning(f"Top strategy exceeds your volatility tolerance by {(current_metrics['volatility'] - profile_data['max_volatility']):.1%}")
                        else:
                            st.success(f"Top strategy aligns with your risk tolerance")
    else:
        st.info("Add at least 2 different assets to get allocation recommendations")
    
    st.divider()
    
    with st.expander("Data Sources & Methodology"):
        st.write("**Data Sources:**")
        st.write("• **Stock Prices**: Yahoo Finance API (real-time)")
        st.write("• **Cryptocurrency Prices**: Binance API via CCXT (real-time)")
        st.write("• **Historical Volatility**: Calculated from historical price data")
        st.write("• **Options Pricing**: Black-Scholes theoretical model")
        st.write("• **Greeks**: Mathematical derivatives of Black-Scholes formula")
        st.write("• **Risk-free Rate**: User input (typically 10-year Treasury rate)")
        
        st.write("**Important Notes:**")
        st.write("• **NO real options market data** - this is theoretical pricing")
        st.write("• **Volatility** is historical stock volatility, not implied volatility")
        st.write("• **Greeks** are calculated, not from options exchanges")
        st.write("• **Contract Size**: 1 options contract = 100 shares (US standard)")
        st.write("• **For real trading**, use actual options prices and implied volatility")
        
        st.write("**Technical Implementation:**")
        st.write("• Black-Scholes pricing with user-defined parameters")
        st.write("• Monte Carlo simulation with correlated asset returns")
        st.write("• Historical VaR from empirical return distribution")
        st.write("• Parametric VaR assuming normal distribution")
    
    st.divider()
    
    st.header("Portfolio Composition")
    
    if len(positions) > 0:
        symbols = [pos.symbol for pos in positions]
        quantities = [abs(pos.quantity) for pos in positions]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(quantities, labels=symbols, autopct='%1.1f%%', startangle=90)
        ax.set_title('Portfolio Composition by Quantity')
        st.pyplot(fig)