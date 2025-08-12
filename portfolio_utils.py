import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import yfinance as yf
from bs_functions import black_scholes_call, black_scholes_puts

try:
    import ccxt
    print(f"CCXT version: {ccxt.__version__}")
    from crypto_utils import fetch_crypto_data, get_crypto_current_price, crypto_fetcher
    CRYPTO_AVAILABLE = True
    print("✅ Crypto functionality enabled")
except ImportError as e:
    CRYPTO_AVAILABLE = False
    print(f"❌ Crypto functionality not available. Install ccxt: pip install ccxt (Error: {e})")
except Exception as e:
    CRYPTO_AVAILABLE = False
    print(f"❌ Error initializing crypto functionality: {e}")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

def fetch_historical_data(symbols, period="1y"):
    try:
        if isinstance(symbols, str):
            symbols = [symbols]
        
        crypto_symbols = []
        stock_symbols = []
        
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.strip():
                symbol = symbol.strip().upper()
                
                if _is_crypto_symbol_simple(symbol):
                    crypto_symbols.append(symbol)
                else:
                    if symbol.startswith('^') or symbol.isalpha():
                        stock_symbols.append(symbol)
                    else:
                        clean_symbol = ''.join(c for c in symbol if c.isalpha()).upper()
                        if clean_symbol:
                            stock_symbols.append(clean_symbol)
        
        all_data = {}
        
        if stock_symbols:
            stock_data = _fetch_stock_data(stock_symbols, period)
            if not stock_data.empty:
                if len(stock_symbols) == 1:
                    all_data[stock_symbols[0]] = stock_data.iloc[:, 0]
                else:
                    for col in stock_data.columns:
                        all_data[col] = stock_data[col]
        
        if crypto_symbols and CRYPTO_AVAILABLE:
            crypto_data = fetch_crypto_data(crypto_symbols, period)
            if not crypto_data.empty:
                for col in crypto_data.columns:
                    all_data[col] = crypto_data[col]
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.DataFrame(all_data)
        return combined_df.dropna()
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def _fetch_stock_data(symbols, period):
    try:
        if len(symbols) == 1:
            ticker = yf.Ticker(symbols[0])
            data = ticker.history(period=period, auto_adjust=True)
            if not data.empty and 'Close' in data.columns:
                result = data[['Close']].dropna().rename(columns={'Close': symbols[0]})
                return result
            else:
                return pd.DataFrame()
        else:
            data = yf.download(symbols, period=period, progress=False, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    result = data['Close'].dropna()
                    return result
                else:
                    return pd.DataFrame()
            else:
                return data.dropna()
                
    except Exception as e:
        return pd.DataFrame()

def calculate_returns(price_data):
    if price_data.empty:
        return pd.DataFrame()
    
    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna()

def calculate_covariance_matrix(returns):
    if returns.empty:
        return pd.DataFrame()
    
    annual_cov_matrix = returns.cov() * 252
    return annual_cov_matrix
    
def historical_var(returns, confidence_level=0.95):
    if len(returns) == 0:
        return 0.0
    
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) == 0:
        return 0.0
    
    var_percentile = (1 - confidence_level) * 100
    var_value = np.percentile(clean_returns, var_percentile)
    return -var_value if var_value<0 else 0.0


def parametric_var(portfolio_value, portfolio_return, portfolio_volatility, confidence_level=0.95, time_horizon=1):
    if portfolio_value <= 0 or portfolio_volatility <= 0:
        return 0.0
    
    z_score = stats.norm.ppf(confidence_level)
    time_factor = np.sqrt(time_horizon/252.0)  # Use trading days
    horizon_return = portfolio_return * (time_horizon / 252.0)
    horizon_volatility = portfolio_volatility * time_factor  # Fixed: was "horizonal_volatility"
    var_return = z_score * horizon_volatility - horizon_return
    var_dollar = portfolio_value * var_return
    return max(var_dollar, 0.0)

def monte_carlo_var(portfolio_positions, num_simulations=10000, confidence_level=0.95, time_horizon=1):
    if not portfolio_positions:
        return 0.0, np.array([]), np.array([]), 0.0
    
    symbols = list(set([pos.symbol for pos in portfolio_positions if hasattr(pos, 'symbol')]))
    current_portfolio_value = sum([pos.get_current_value() for pos in portfolio_positions])
    
    if not symbols:
        simulated_returns = np.random.normal(0, 0.02, num_simulations)
        simulated_portfolio_values = current_portfolio_value * (1 + simulated_returns)
        var_value = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return -var_value * current_portfolio_value, simulated_portfolio_values, simulated_returns.reshape(-1, 1), current_portfolio_value
    
    try:
        price_data = fetch_historical_data(symbols, period="1y")
        if price_data.empty:
            simulated_returns = np.random.normal(0, 0.02, num_simulations)
            simulated_portfolio_values = current_portfolio_value * (1 + simulated_returns)
            var_value = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            return -var_value * current_portfolio_value, simulated_portfolio_values, simulated_returns.reshape(-1, 1), current_portfolio_value
        
        returns = calculate_returns(price_data)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        if time_horizon == 1:
            scaled_mean = mean_returns
            scaled_cov = cov_matrix
        else:
            time_factor = time_horizon  # Direct scaling for multi-day
            scaled_mean = mean_returns * time_factor
            scaled_cov = cov_matrix * time_factor
        
        current_prices = {}
        for symbol in symbols:
            try:
                if _is_crypto_symbol_simple(symbol) and CRYPTO_AVAILABLE:
                    crypto_price = get_crypto_current_price(symbol)
                    if crypto_price:
                        current_prices[symbol] = float(crypto_price)
                        print(f"💰 {symbol}: ${crypto_price:,.2f} (crypto)")
                    else:
                        current_prices[symbol] = 100.0
                else:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", auto_adjust=True)
                    if not data.empty and 'Close' in data.columns:
                        close_price = data['Close'].iloc[-1]
                        current_prices[symbol] = float(close_price)
                        print(f"📈 {symbol}: ${close_price:.2f} (stock)")
                    else:
                        current_prices[symbol] = 100.0
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")
                current_prices[symbol] = 100.0
        
        if NUMBA_AVAILABLE and num_simulations >= 1000:
            simulated_portfolio_values = _monte_carlo_jit(portfolio_positions, symbols, current_prices, scaled_mean, scaled_cov, num_simulations)
        else:
            simulated_portfolio_values = _monte_carlo_python(portfolio_positions, symbols, current_prices, scaled_mean, scaled_cov, num_simulations)
        
        if len(symbols) == 1:
            simulated_asset_returns = np.random.normal(scaled_mean[0], np.sqrt(scaled_cov[0, 0]), num_simulations).reshape(-1, 1)
        else:
            simulated_asset_returns = np.random.multivariate_normal(scaled_mean, scaled_cov, num_simulations)
        
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(simulated_portfolio_values, var_percentile)
        var_dollar = abs(current_portfolio_value - var_value)
        
        return var_dollar, simulated_portfolio_values, simulated_asset_returns, current_portfolio_value
        
    except Exception as e:
        simulated_returns = np.random.normal(0, 0.02, num_simulations)
        simulated_portfolio_values = current_portfolio_value * (1 + simulated_returns)
        var_value = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return -var_value * current_portfolio_value, simulated_portfolio_values, simulated_returns.reshape(-1, 1), current_portfolio_value

def _monte_carlo_jit(portfolio_positions, symbols, current_prices, scaled_mean, scaled_cov, num_simulations):
    try:
        n_assets = len(symbols)
        symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}
        
        quantities = []
        strikes = []
        position_types = []  # 0-stock, 1-call, 2-put
        asset_indices = []
        
        for pos in portfolio_positions:
            if pos.symbol in symbol_to_index:
                quantities.append(float(pos.quantity))
                asset_indices.append(symbol_to_index[pos.symbol])
                
                if pos.instrument_type == 'stock':
                    position_types.append(0)
                    strikes.append(0.0)  # Not used for stocks
                elif pos.instrument_type == 'call_option':
                    position_types.append(1)
                    strikes.append(float(pos.strike))
                elif pos.instrument_type == 'put_option':
                    position_types.append(2)
                    strikes.append(float(pos.strike))
                else:
                    position_types.append(0)  # Default is stock
                    strikes.append(0.0)
        
        quantities = np.array(quantities)
        strikes = np.array(strikes)
        position_types = np.array(position_types, dtype=np.int32)
        asset_indices = np.array(asset_indices, dtype=np.int32)
        current_prices_array = np.array([current_prices[symbol] for symbol in symbols])
        
        if n_assets == 1:
            random_returns = np.random.normal(scaled_mean[0], np.sqrt(scaled_cov[0, 0]), num_simulations).reshape(-1, 1)
        else:
            chol_matrix = np.linalg.cholesky(scaled_cov)
            random_normals = np.random.standard_normal((num_simulations, n_assets))
            random_returns = jit_generate_correlated_returns(scaled_mean, chol_matrix, random_normals)
        
        future_prices = current_prices_array * np.exp(random_returns)
        
        portfolio_values = jit_portfolio_valuation(future_prices[:, asset_indices], quantities, strikes, position_types)
        return portfolio_values
        
    except Exception as e:
        return _monte_carlo_python(portfolio_positions, symbols, current_prices, scaled_mean, scaled_cov, num_simulations)

def _monte_carlo_python(portfolio_positions, symbols, current_prices, scaled_mean, scaled_cov, num_simulations):
    if len(symbols) == 1:
        simulated_asset_returns = np.random.normal(scaled_mean[0], np.sqrt(scaled_cov[0, 0]), num_simulations).reshape(-1, 1)
    else:
        simulated_asset_returns = np.random.multivariate_normal(scaled_mean, scaled_cov, num_simulations)
    
    simulated_portfolio_values = []
    
    for sim in range(num_simulations):
        portfolio_value = 0
        future_prices = {}
        for i, symbol in enumerate(symbols):
            future_prices[symbol] = current_prices[symbol] * np.exp(simulated_asset_returns[sim, i])
        
        for position in portfolio_positions:
            if position.instrument_type in ['stock', 'cryptocurrency']:
                if position.symbol in future_prices:
                    position_value = position.quantity * future_prices[position.symbol]
                else:
                    position_value = position.get_current_value()
                    
            elif 'option' in position.instrument_type:
                if position.symbol in future_prices:
                    future_spot = future_prices[position.symbol]
                    
                    try:
                        if 'option' in position.instrument_type:
                            time_decay = time_horizon / 365.0
                            adjusted_time_to_expiry = max(0.001, position.time_to_expiry - time_decay)
                        else:
                            adjusted_time_to_expiry = position.time_to_expiry
                        
                        if position.instrument_type == 'call_option':
                            option_price = black_scholes_call(future_spot, position.strike, adjusted_time_to_expiry,position.risk_free_rate, position.volatility)
                        else:
                            option_price = black_scholes_puts(future_spot, position.strike, adjusted_time_to_expiry, position.risk_free_rate, position.volatility)
                        
                        position_value = position.quantity * option_price * 100
                    except:
                        if position.instrument_type == 'call_option':
                            option_payoff = max(0, future_spot - position.strike)
                        else:
                            option_payoff = max(0, position.strike - future_spot)
                        position_value = position.quantity * option_payoff * 100
                else:
                    position_value = position.get_current_value()
            else:
                position_value = position.get_current_value()
            
            portfolio_value += position_value
        
        simulated_portfolio_values.append(portfolio_value)
    
    return np.array(simulated_portfolio_values)


def calculate_cvar(returns, confidence_level=0.95):
    if len(returns) == 0:
        return 0.0
    
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) == 0:
        return 0.0
    
    var_percentile = (1 - confidence_level) * 100
    var_threshold = np.percentile(clean_returns, var_percentile)
    tail_losses = clean_returns[clean_returns <= var_threshold]

    if len(tail_losses) == 0:
        return -var_threshold if var_threshold < 0 else 0.0
    
    cvar_value = np.mean(tail_losses)
    return -cvar_value if cvar_value < 0 else 0.0

def stress_test_scenarios(): # examples to pull from
    scenarios = {
        "2008_financial_crisis": {
            "equity_shock": -0.37,
            "vol_increase": 1.5,
            "rate_change": -0.03
        },
        "covid_crash_2020": {
            "equity_shock": -0.34,
            "vol_increase": 2.0,
            "rate_change": -0.015
        },
        "dot_com_bubble": {
            "equity_shock": -0.49,
            "vol_increase": 1.3,
            "rate_change": -0.025
        }
    }
    
    return scenarios

def apply_stress_scenario(portfolio_positions, scenario):
    if not portfolio_positions:
        return 0.0
    
    try:
        current_value = sum([pos.get_current_value() for pos in portfolio_positions])
        stressed_value = 0.0
        
        for position in portfolio_positions:
            if position.instrument_type == 'stock':
                equity_shock = scenario.get('equity_shock', 0.0)
                current_price = position._get_current_spot_price()
                stressed_price = current_price * (1 + equity_shock)
                stressed_position_value = position.quantity * stressed_price
                
            elif 'option' in position.instrument_type:
                equity_shock = scenario.get('equity_shock', 0.0)
                vol_increase = scenario.get('vol_increase', 1.0)
                rate_change = scenario.get('rate_change', 0.0)
                stressed_spot = position.spot_price * (1 + equity_shock)
                stressed_vol = position.volatility * vol_increase
                stressed_rate = position.risk_free_rate + rate_change
                if position.instrument_type == 'call_option':
                    stressed_option_price = black_scholes_call(stressed_spot, position.strike, position.time_to_expiry, stressed_rate, stressed_vol)
                else:
                    stressed_option_price = black_scholes_puts(stressed_spot, position.strike, position.time_to_expiry,stressed_rate, stressed_vol)
                
                stressed_position_value = position.quantity * stressed_option_price * 100
            
            else:
                stressed_position_value = position.get_current_value()
            
            stressed_value += stressed_position_value
        
        stress_impact = stressed_value - current_value
        return stress_impact
        
    except Exception as e:
        print(f"Error applying stress scenario: {e}")
        return 0.0

def calculate_portfolio_beta(portfolio_weights, individual_betas):
    return np.sum(portfolio_weights * individual_betas)

def calculate_maximum_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return abs(max_drawdown)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    annual_return = np.mean(returns) * 252
    annual_volatility = np.std(returns) * np.sqrt(252)
    profit = annual_return - risk_free_rate
    sharpe_ratio = profit / annual_volatility
    return sharpe_ratio

@jit(nopython=True, cache=True)
def jit_option_payoff(spot_prices, strikes, is_call):
    n = len(spot_prices)
    payoffs = np.zeros(n)
    
    for i in prange(n):
        if is_call:
            payoffs[i] = max(0.0, spot_prices[i] - strikes[i])
        else:
            payoffs[i] = max(0.0, strikes[i] - spot_prices[i])
    
    return payoffs

@jit(nopython=True, cache=True)
def jit_portfolio_valuation(future_prices, quantities, strikes, position_types, contracts_multiplier=100.0):
    n_positions = len(quantities)
    n_simulations = future_prices.shape[0]
    portfolio_values = np.zeros(n_simulations)
    
    for sim in prange(n_simulations):
        portfolio_value = 0.0
        
        for pos in range(n_positions):
            if position_types[pos] == 0:  # Stock
                position_value = quantities[pos] * future_prices[sim, pos]
            elif position_types[pos] == 1:  # Call option
                option_payoff = max(0.0, future_prices[sim, pos] - strikes[pos])
                position_value = quantities[pos] * option_payoff * contracts_multiplier
            elif position_types[pos] == 2:  # Put option
                option_payoff = max(0.0, strikes[pos] - future_prices[sim, pos])
                position_value = quantities[pos] * option_payoff * contracts_multiplier
            else:
                position_value = 0.0
            
            portfolio_value += position_value
        
        portfolio_values[sim] = portfolio_value
    
    return portfolio_values

@jit(nopython=True, cache=True)
def jit_generate_correlated_returns(mean_returns, chol_matrix, random_normals):
    n_assets = len(mean_returns)
    n_simulations = random_normals.shape[0]
    correlated_returns = np.zeros((n_simulations, n_assets))
    
    for sim in prange(n_simulations):
        uncorrelated = random_normals[sim, :]
        correlated = np.dot(chol_matrix, uncorrelated)
        
        for asset in range(n_assets):
            correlated_returns[sim, asset] = mean_returns[asset] + correlated[asset]
    
    return correlated_returns

def create_correlation_heatmap(symbols, period="1y"):
    try:
        price_data = fetch_historical_data(symbols, period=period)
        
        if price_data.empty or len(symbols) < 2:
            return None, None
        
        returns = calculate_returns(price_data)
        
        if returns.empty:
            return None, None
        
        correlation_matrix = returns.corr()
        
        return correlation_matrix, symbols
    except Exception as e:
        return None, None

def analyze_monte_carlo_results(simulated_values, current_value, time_horizon=1, confidence_levels=[0.90, 0.95, 0.99]):
    if len(simulated_values) == 0:
        return {}
    
    returns = (simulated_values - current_value) / current_value
    annual_scale = 252 / time_horizon  # 252 trading days per year
    
    results = {
        'current_value': current_value,
        'mean_simulated_value': np.mean(simulated_values),
        'std_simulated_value': np.std(simulated_values),
        'min_value': np.min(simulated_values),
        'max_value': np.max(simulated_values),
        'mean_return': np.mean(returns),
        'volatility': np.std(returns),
        'annual_mean_return': np.mean(returns) * annual_scale,
        'annual_volatility': np.std(returns) * np.sqrt(annual_scale),
        'skewness': float(pd.Series(returns).skew()),
        'kurtosis': float(pd.Series(returns).kurtosis()),
        'time_horizon': time_horizon,
    }
    
    for conf_level in confidence_levels:
        var_percentile = (1 - conf_level) * 100
        var_value = np.percentile(simulated_values, var_percentile)
        var_loss = current_value - var_value
        
        tail_values = simulated_values[simulated_values <= var_value]
        if len(tail_values) > 0:
            cvar_value = np.mean(tail_values)
            cvar_loss = current_value - cvar_value
        else:
            cvar_loss = var_loss
        
        results[f'var_{int(conf_level*100)}'] = max(0, var_loss)
        results[f'cvar_{int(conf_level*100)}'] = max(0, cvar_loss)
    
    return results

def _is_crypto_symbol_simple(symbol):
    crypto_indicators = ['/USDT', '/BUSD', '/BTC', '/ETH', '/BNB']
    return any(indicator in symbol.upper() for indicator in crypto_indicators)