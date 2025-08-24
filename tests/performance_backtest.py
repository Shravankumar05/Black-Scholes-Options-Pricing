import numpy as np
import pandas as pd
import yfinance as yf
from portfolio_allocation import EnhancedPortfolioAllocationEngine

def run_backtest(engine, assets, start_date, end_date, rebalance_freq='M'):
    """Runs a historical backtest for a given portfolio allocation engine."""
    print(f"🚀 Starting backtest for assets: {assets}")
    print(f"Period: {start_date} to {end_date}\n")

    # Download historical data
    price_data = yf.download(assets, start=start_date, end=end_date)['Close']
    returns_data = price_data.pct_change().dropna()

    # Get rebalancing dates
    rebalance_dates = price_data.resample(rebalance_freq).first().index

    # Backtest loop
    portfolio_value = [100.0]
    current_weights = {asset: 1.0/len(assets) for asset in assets}
    last_rebalance_date = returns_data.index[0]

    for date in returns_data.index:
        if date in rebalance_dates and date > last_rebalance_date:
            print(f"\n🔄 Rebalancing on {date.date()}...")
            # Get historical data up to the rebalancing day
            historic_returns = returns_data.loc[:date]
            if len(historic_returns) > 60: # Need enough data for the engine
                try:
                    current_weights = engine.regime_adaptive_allocation(historic_returns, assets)
                    print(f"   New weights: { {k: f'{v:.2%}' for k, v in current_weights.items()} }")
                except Exception as e:
                    print(f"   ⚠️ Allocation failed: {e}. Holding previous weights.")
            last_rebalance_date = date

        # Calculate daily portfolio return
        daily_returns = returns_data.loc[date]
        portfolio_return = sum(current_weights.get(asset, 0) * daily_returns.get(asset, 0) for asset in assets)
        portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))

    # Create portfolio performance DataFrame
    portfolio_df = pd.DataFrame(portfolio_value[1:], index=returns_data.index, columns=['Portfolio_Value'])
    return portfolio_df

def calculate_performance_metrics(portfolio_df):
    """Calculates key performance metrics from a portfolio value series."""
    metrics = {}
    returns = portfolio_df['Portfolio_Value'].pct_change().dropna()

    # Total Return
    metrics['Total Return'] = (portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0] - 1)

    # Annualized Return
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (365.0/days) - 1

    # Annualized Volatility
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)

    # Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    metrics['Sharpe Ratio'] = (metrics['Annualized Return'] - risk_free_rate) / metrics['Annualized Volatility']

    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['Max Drawdown'] = drawdown.min()

    # Calmar Ratio
    metrics['Calmar Ratio'] = metrics['Annualized Return'] / abs(metrics['Max Drawdown'])

    return metrics

if __name__ == '__main__':
    # --- Configuration ---
    ASSETS_ETF = ['SPY', 'AGG', 'GLD', 'QQQ', 'EFA'] # Diversified ETFs
    START_DATE = '2015-01-01'
    END_DATE = '2023-12-31'

    # --- Run Backtest ---
    allocation_engine = EnhancedPortfolioAllocationEngine()
    portfolio_performance = run_backtest(allocation_engine, ASSETS_ETF, START_DATE, END_DATE)

    # --- Calculate and Print Metrics ---
    performance_metrics = calculate_performance_metrics(portfolio_performance)

    print("\n\n--- Backtest Results ---", flush=True)
    for metric, value in performance_metrics.items():
        print(f"{metric+':':<25} {value:.2% if '%' in metric or 'Return' in metric or 'Ratio' in metric else value:.4f}", flush=True)
    print("------------------------", flush=True)

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        portfolio_performance['Portfolio_Value'].plot(ax=ax, title='Portfolio Performance Over Time')
        ax.set_ylabel('Portfolio Value (Indexed to 100)')
        ax.set_xlabel('Date')
        plt.savefig('performance_chart.png')
        print("\nChart saved to performance_chart.png")
    except ImportError:
        print("\nMatplotlib not found. Skipping plot. Install with: pip install matplotlib")
