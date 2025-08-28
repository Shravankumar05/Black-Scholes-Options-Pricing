# Advanced Portfolio Analysis & ML-Driven Backtesting Engine

## Key Project Highlights

- **Interactive Dashboard**: A user-friendly interface for real-time Black-Scholes options pricing, Greek analysis, and portfolio risk management.
- **ML-Powered Backtesting**: An advanced backtesting engine that uses regime detection and ML-enhanced models to dynamically adjust portfolio allocations.
- **Comprehensive Risk Management**: Implements a wide range of risk metrics, including various VaR methods, CVaR, and historical stress testing.
- **Performance Optimized**: Utilizes Numba for JIT compilation and parallel processing to accelerate complex financial calculations.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shravankumar05/Black-Scholes-Options-Pricing.git
    cd Black-Scholes-Options-Pricing
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Project Demo

This project has two main components that can be demonstrated:

### 1. Interactive Portfolio Analysis Dashboard

The Streamlit application provides a GUI for real-time options and portfolio analysis.

**To run the dashboard:**

```bash
streamlit run bs_app.py
```

This will launch a web server and open the dashboard in your browser, where you can:

-   Price individual options using the Black-Scholes model.
-   Analyze option Greeks and visualize P&L scenarios with heatmaps.
-   Construct multi-asset portfolios and evaluate their risk profiles.

### 2. ML-Driven Performance Backtest

The backtesting engine evaluates the performance of the machine learning-enhanced portfolio allocation strategies.

**To run the backtest:**

```bash
python performance_backtest.py
```

This script will:

-   Fetch historical market data for a predefined set of assets.
-   Train the ML models for market regime detection and return prediction.
-   Run a historical simulation of the portfolio's performance.
-   Output performance metrics and generate a `performance_chart.png` visualizing the results.

## ✨ Features & Technologies

### Core Technologies

-   **Python**: The core programming language.
-   **Streamlit**: For building the interactive web dashboard.
-   **Scikit-learn**: For machine learning models (Random Forest, GMM).
-   **Pandas & NumPy**: For data manipulation and numerical computation.
-   **SciPy**: For optimization and statistical functions.
-   **Matplotlib & Seaborn**: For data visualization.
-   **Numba**: For JIT compilation to accelerate numerical functions.

### Feature Breakdown

#### Portfolio Allocation Engine (`portfolio_allocation.py`)

-   **Market Regime Detection**: Uses a Gaussian Mixture Model (GMM) to classify market conditions (e.g., Bull, Bear, High-Volatility) based on features like volatility, correlation, and momentum.
-   **ML-Enhanced Return Prediction**: Employs a Random Forest Regressor to predict future asset returns, which informs the allocation strategy.
-   **Dynamic Allocation Strategies**: Implements a range of allocation models that adapt to the current market regime, including:
    -   ML-Enhanced Black-Litterman
    -   Advanced Minimum Variance
    -   Regime-Adaptive Volatility Targeting

#### Performance Backtesting (`performance_backtest.py`)

-   **Historical Simulation**: Simulates portfolio performance over a specified historical period.
-   **Performance Metrics**: Calculates key metrics such as Annualized Return, Volatility, Sharpe Ratio, and Max Drawdown.
-   **Visualization**: Generates charts comparing the portfolio's performance against a benchmark.

#### Interactive Dashboard (`bs_app.py`)

-   **Black-Scholes Model**: Provides real-time pricing for call and put options.
-   **Greeks Analysis**: Calculates and displays Delta, Gamma, Theta, Vega, and Rho.
-   **Portfolio Risk Management**: Offers tools for VaR, CVaR, and stress testing on user-defined portfolios.

## 🔧 Project Architecture

-   `bs_app.py`: The entry point for the Streamlit interactive dashboard.
-   `performance_backtest.py`: The entry point for running the ML-driven backtest.
-   `portfolio_allocation.py`: Contains the core logic for the advanced portfolio allocation engine, including ML models and allocation strategies.
-   `bs_functions.py`: Implements the Black-Scholes pricing model and Greeks calculations.
-   `ml_components.py`, `crypto_utils.py`, `db_utils.py`: Utility modules for machine learning components, cryptocurrency data, and database interactions.