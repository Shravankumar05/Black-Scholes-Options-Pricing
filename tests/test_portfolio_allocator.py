import numpy as np
import pandas as pd
import yfinance as yf
from portfolio_allocation import portfolio_allocator
import warnings
warnings.filterwarnings('ignore')

def generate_portfolio_data(assets, days=252):
    """Generate synthetic portfolio data for testing"""
    np.random.seed(42)
    
    # Generate correlated returns
    n_assets = len(assets)
    
    # Create correlation matrix (more realistic)
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(0.1, 0.6)  # Positive correlations
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Generate returns with different characteristics for each asset
    mean_returns = np.random.uniform(0.05, 0.25, n_assets)  # Annual returns 5-25%
    volatilities = np.random.uniform(0.1, 0.4, n_assets)    # Annual vol 10-40%
    
    # Generate correlated returns
    chol_matrix = np.linalg.cholesky(corr_matrix)
    uncorrelated_returns = np.random.normal(0, 1, (days, n_assets))
    correlated_returns = uncorrelated_returns @ chol_matrix.T
    
    # Scale to desired mean and volatility
    returns_matrix = np.zeros((days, n_assets))
    for i in range(n_assets):
        returns_matrix[:, i] = correlated_returns[:, i] * volatilities[i] / np.sqrt(252) + mean_returns[i] / 252
    
    # Create price series
    prices = np.zeros((days, n_assets))
    initial_prices = np.random.uniform(50, 200, n_assets)  # Initial prices $50-200
    prices[0, :] = initial_prices
    
    for t in range(1, days):
        prices[t, :] = prices[t-1, :] * (1 + returns_matrix[t, :])
    
    # Create DataFrame
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    price_data = pd.DataFrame(prices, columns=assets, index=dates)
    
    return price_data

def evaluate_allocation_strategies(assets, risk_profile='moderate'):
    """Evaluate all allocation strategies"""
    print(f"=== Portfolio Allocation Evaluation ({risk_profile.title()} Risk Profile) ===")
    
    # Generate test data
    price_data = generate_portfolio_data(assets, days=500)
    returns_data = price_data.pct_change().dropna()
    
    print(f"Generated data for {len(assets)} assets over {len(returns_data)} days")
    print(f"Average returns: {returns_data.mean().mean()*252:.2%}")
    print(f"Average volatility: {returns_data.std().mean()*np.sqrt(252):.2%}")
    
    # Get allocation recommendations
    recommendations = portfolio_allocator.get_allocation_recommendation(
        returns_data=returns_data,
        assets=assets,
        risk_profile=risk_profile,
        market_regime='normal'
    )
    
    # Evaluate each strategy
    results = []
    for strategy_name, strategy_data in recommendations['recommendations'].items():
        metrics = strategy_data['metrics']
        suitability = strategy_data['suitability_score']
        
        # Calculate additional metrics
        weights = strategy_data['allocation']
        asset_weights = np.array([weights.get(asset, 0) for asset in assets])
        
        # Concentration measure (Herfindahl-Hirschman Index)
        hhi = np.sum(asset_weights ** 2)
        concentration = "High" if hhi > 0.3 else "Medium" if hhi > 0.15 else "Low"
        
        results.append({
            'Strategy': strategy_name,
            'Return': metrics['return'],
            'Volatility': metrics['volatility'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Suitability Score': suitability,
            'Concentration': concentration,
            'HHI': hhi
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Suitability Score', ascending=False)
    
    print("\n=== Strategy Performance Rankings ===")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Analyze top strategy
    top_strategy = recommendations['top_recommendation']
    if top_strategy:
        strategy_name, strategy_data = top_strategy
        print(f"\n=== Top Strategy: {strategy_name} ===")
        print(f"Suitability Score: {strategy_data['suitability_score']:.2f}")
        print(f"Expected Return: {strategy_data['metrics']['return']:.2%}")
        print(f"Expected Volatility: {strategy_data['metrics']['volatility']:.2%}")
        print(f"Sharpe Ratio: {strategy_data['metrics']['sharpe_ratio']:.4f}")
        
        # Show allocation
        print("\nAllocation Breakdown:")
        allocation = strategy_data['allocation']
        for asset, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01:  # Show only allocations > 1%
                print(f"  {asset}: {weight:.1%}")
    
    return results_df, recommendations

def test_risk_profiles(assets):
    """Test allocation strategies across different risk profiles"""
    print("\n=== Risk Profile Comparison ===")
    
    risk_profiles = ['conservative', 'moderate', 'aggressive']
    profile_results = {}
    
    for profile in risk_profiles:
        price_data = generate_portfolio_data(assets, days=500)
        returns_data = price_data.pct_change().dropna()
        
        recommendations = portfolio_allocator.get_allocation_recommendation(
            returns_data=returns_data,
            assets=assets,
            risk_profile=profile,
            market_regime='normal'
        )
        
        top_strategy = recommendations['top_recommendation']
        if top_strategy:
            _, strategy_data = top_strategy
            profile_results[profile] = {
                'return': strategy_data['metrics']['return'],
                'volatility': strategy_data['metrics']['volatility'],
                'sharpe': strategy_data['metrics']['sharpe_ratio']
            }
    
    # Display comparison
    comparison_df = pd.DataFrame(profile_results).T
    print(comparison_df.to_string(float_format='%.4f'))

def test_rebalancing_suggestions(assets):
    """Test rebalancing suggestions functionality"""
    print("\n=== Rebalancing Suggestions Test ===")
    
    # Generate current allocations (random)
    np.random.seed(123)
    current_weights = np.random.dirichlet(np.ones(len(assets)), size=1)[0]
    current_allocations = dict(zip(assets, current_weights))
    
    # Generate target allocations (different random)
    np.random.seed(456)
    target_weights = np.random.dirichlet(np.ones(len(assets)), size=1)[0]
    target_allocations = dict(zip(assets, target_weights))
    
    print("Current Allocations:")
    for asset, weight in sorted(current_allocations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset}: {weight:.1%}")
    
    print("\nTarget Allocations:")
    for asset, weight in sorted(target_allocations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset}: {weight:.1%}")
    
    # Get rebalancing suggestions
    suggestions = portfolio_allocator.rebalancing_suggestions(
        current_allocations, target_allocations, threshold=0.05
    )
    
    print(f"\nRebalancing Suggestions ({len(suggestions)} changes needed):")
    if suggestions:
        for suggestion in suggestions:
            action = suggestion['action']
            asset = suggestion['asset']
            current = suggestion['current_weight']
            target = suggestion['target_weight']
            diff = suggestion['difference']
            print(f"  {action} {asset}: {current:.1%} → {target:.1%} (Δ: {diff:+.1%})")
    else:
        print("  No significant rebalancing needed (all differences < 5%)")

def stress_test_allocator(assets):
    """Stress test the allocator with extreme market conditions"""
    print("\n=== Stress Testing ===")
    
    # Test with highly volatile assets
    print("Testing with high volatility scenario...")
    price_data = generate_portfolio_data(assets, days=500)
    # Multiply returns by 3 to simulate high volatility
    returns_data = price_data.pct_change().dropna() * 3
    
    recommendations = portfolio_allocator.get_allocation_recommendation(
        returns_data=returns_data,
        assets=assets,
        risk_profile='moderate',
        market_regime='high_volatility'
    )
    
    top_strategy = recommendations['top_recommendation']
    if top_strategy:
        _, strategy_data = top_strategy
        print(f"High volatility scenario - Sharpe Ratio: {strategy_data['metrics']['sharpe_ratio']:.4f}")
        print(f"Volatility: {strategy_data['metrics']['volatility']:.2%}")
    
    # Test with low correlation assets
    print("Testing with low correlation scenario...")
    price_data_low_corr = generate_portfolio_data(assets, days=500)
    # Reduce correlations
    returns_data_low_corr = price_data_low_corr.pct_change().dropna()
    # Add noise to reduce correlations further
    noise = np.random.normal(0, 0.01, returns_data_low_corr.shape)
    returns_data_low_corr = returns_data_low_corr + noise
    
    recommendations_low_corr = portfolio_allocator.get_allocation_recommendation(
        returns_data=returns_data_low_corr,
        assets=assets,
        risk_profile='moderate',
        market_regime='normal'
    )
    
    top_strategy_low_corr = recommendations_low_corr['top_recommendation']
    if top_strategy_low_corr:
        _, strategy_data = top_strategy_low_corr
        print(f"Low correlation scenario - Sharpe Ratio: {strategy_data['metrics']['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    print("Testing Portfolio Allocation Engine")
    print("=" * 50)
    
    # Test assets
    test_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Evaluate allocation strategies
    results_df, recommendations = evaluate_allocation_strategies(test_assets, 'moderate')
    
    # Test different risk profiles
    test_risk_profiles(test_assets)
    
    # Test rebalancing suggestions
    test_rebalancing_suggestions(test_assets)
    
    # Stress test
    stress_test_allocator(test_assets)
    
    # Performance summary
    print("\n=== Performance Summary ===")
    avg_sharpe = results_df['Sharpe Ratio'].mean()
    best_strategy = results_df.loc[results_df['Sharpe Ratio'].idxmax(), 'Strategy']
    best_sharpe = results_df['Sharpe Ratio'].max()
    
    print(f"Average Sharpe Ratio across strategies: {avg_sharpe:.4f}")
    print(f"Best performing strategy: {best_strategy} (Sharpe: {best_sharpe:.4f})")
    
    if avg_sharpe > 0.5:
        print("✅ Portfolio allocator performance is EXCELLENT (Avg Sharpe > 0.5)")
    elif avg_sharpe > 0.2:
        print("⚠️ Portfolio allocator performance is GOOD (0.2 < Avg Sharpe < 0.5)")
    else:
        print("❌ Portfolio allocator performance is POOR (Avg Sharpe < 0.2)")
        print("Consider implementing additional strategies or improving existing ones")
    
    print("\n=== Test Complete ===")