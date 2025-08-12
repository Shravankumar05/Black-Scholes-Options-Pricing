# Black-Scholes Options Analysis Dashboard

## 🚀 Features

### Individual Options Analysis
- **Black-Scholes Pricing Model**: Real-time options pricing for calls and puts
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, and Rho analysis
- **Implied Volatility**: Newton-Raphson method for IV calculation
- **Interactive Heatmaps**: Price and P&L visualization across spot/volatility ranges
- **Break-Even Analysis**: Visual break-even curves for option strategies
- **Live Market Data**: Real-time price and volatility fetching via yfinance and Binance

### Portfolio Risk Management
- **Multi-Asset Portfolios**: Support for stocks, cryptocurrencies, calls, and puts in single portfolio
- **Value at Risk (VaR)**: Historical, Parametric, and Monte Carlo VaR methods
- **Conditional VaR (CVaR)**: Expected Shortfall analysis for tail risk
- **Monte Carlo Simulation**: JIT-optimized simulations with up to 10,000 scenarios
- **Correlation Analysis**: Asset correlation matrices with visualization
- **Stress Testing**: Historical scenario analysis (2008 Crisis, COVID, Dot-com)
- **Portfolio Greeks**: Aggregated risk sensitivities across all positions
- **Cryptocurrency Support**: Real-time crypto data via Binance API

### Advanced Technical Features
- **Performance Optimization**: Numba JIT compilation for Monte Carlo simulations
- **Multi-threading**: Automatic parallelization for large calculations
- **Real-time Data**: Integration with Yahoo Finance API
- **Database Integration**: SQLite storage for calculation history
- **Professional Visualizations**: Publication-ready charts and heatmaps

## 📊 Risk Metrics Implemented

### Value at Risk (VaR)
- **Historical VaR**: Non-parametric approach using empirical distribution
- **Parametric VaR**: Variance-covariance method assuming normal distribution
- **Monte Carlo VaR**: Simulation-based approach handling complex portfolios

### Advanced Risk Measures
- **Conditional VaR (CVaR)**: Average loss beyond VaR threshold
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Portfolio Beta**: Systematic risk measurement

### Stress Testing
- **Historical Scenarios**: 2008 Financial Crisis, COVID-19 Crash, Dot-com Bubble
- **Hypothetical Shocks**: Custom equity, volatility, and interest rate scenarios
- **Correlation Breakdown**: Crisis correlation modeling

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shravankumar05/Black-Scholes-Options-Pricing.git
cd Black-Scholes-Options-Pricing
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run bs_app.py
```

## 💻 Usage

### Individual Options Analysis
1. Select "Individual Black-Scholes" from the sidebar
2. Enter stock symbol (optional) for live data
3. Configure option parameters (strike, expiry, volatility, etc.)
4. View real-time pricing, Greeks, and risk analysis
5. Generate interactive heatmaps and break-even curves

### Portfolio Risk Analysis
1. Select "Portfolio Risk Analysis" from the sidebar
2. Add multiple positions (stocks, calls, puts)
3. Configure risk parameters (confidence levels, time horizons)
4. Run Monte Carlo simulations and stress tests
5. Analyze correlation matrices and portfolio Greeks

## 🔧 Technical Architecture

### Core Components
- **bs_functions.py**: Black-Scholes pricing and Greeks calculations
- **portfolio_utils.py**: Risk management and Monte Carlo functions
- **portfolio_risk.py**: Portfolio analysis UI and logic
- **db_utils.py**: Database operations and data persistence

### Performance Features
- **JIT Compilation**: Numba-optimized Monte Carlo simulations
- **Vectorization**: NumPy-based calculations for speed
- **Multi-threading**: Parallel processing for large datasets
- **Caching**: Intelligent caching for repeated calculations

### Data Sources
- **Yahoo Finance**: Real-time stock market data and historical prices
- **Binance API**: Real-time cryptocurrency data and historical prices
- **User Input**: Custom parameters and scenarios
- **SQLite Database**: Calculation history and results storage

## 📈 Mathematical Models

### Black-Scholes Formula
```
Call: C = S₀N(d₁) - Ke^(-rT)N(d₂)
Put:  P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Greeks Calculations
- **Delta**: ∂V/∂S (price sensitivity)
- **Gamma**: ∂²V/∂S² (delta sensitivity)
- **Theta**: ∂V/∂T (time decay)
- **Vega**: ∂V/∂σ (volatility sensitivity)
- **Rho**: ∂V/∂r (interest rate sensitivity)

### Risk Metrics
- **VaR**: Percentile-based loss estimation
- **CVaR**: E[Loss | Loss > VaR]
- **Monte Carlo**: Correlated asset simulation using Cholesky decomposition