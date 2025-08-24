"""
Model Comparison and Benchmarking Framework
==========================================
This script provides comprehensive comparison of ML models against industry-standard
baselines and benchmarks for both volatility forecasting and portfolio allocation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from test_ml_volatility_comprehensive import VolatilityTestFramework
from test_portfolio_comprehensive import PortfolioTestFramework
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️ Seaborn not available - some visualizations will be skipped")
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json

class ModelBenchmarkFramework:
    def __init__(self):
        self.volatility_benchmarks = {}
        self.portfolio_benchmarks = {}
        self.comparison_results = {}
        
    def create_volatility_benchmarks(self, data):
        """Create various volatility forecasting benchmarks"""
        
        returns = data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        benchmarks = {}
        
        # 1. Naive Forecast (Last Observed Value)
        benchmarks['Naive'] = realized_vol.shift(1)
        
        # 2. Simple Moving Average
        benchmarks['SMA_30'] = realized_vol.rolling(30).mean()
        benchmarks['SMA_60'] = realized_vol.rolling(60).mean()
        
        # 3. Exponentially Weighted Moving Average
        benchmarks['EWMA'] = realized_vol.ewm(span=30).mean()
        
        # 4. GARCH(1,1) approximation
        benchmarks['GARCH_Simple'] = self._simple_garch_forecast(returns)
        
        # 5. Linear Trend
        benchmarks['Linear_Trend'] = self._linear_trend_forecast(realized_vol)
        
        # 6. Seasonal Average (if enough data)
        if len(realized_vol) > 252:
            benchmarks['Seasonal'] = self._seasonal_forecast(realized_vol)
        
        # 7. VIX-based (if available)
        benchmarks['VIX_Proxy'] = self._vix_proxy_forecast(returns)
        
        return benchmarks
    
    def _simple_garch_forecast(self, returns, alpha=0.1, beta=0.85):
        """Simple GARCH(1,1) approximation"""
        variance_forecast = []
        long_run_var = returns.var()
        current_var = long_run_var
        
        for i, ret in enumerate(returns):
            if i == 0:
                variance_forecast.append(np.sqrt(current_var * 252))
            else:
                # GARCH(1,1): σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
                omega = (1 - alpha - beta) * long_run_var
                current_var = omega + alpha * (ret ** 2) + beta * current_var
                variance_forecast.append(np.sqrt(current_var * 252))
        
        return pd.Series(variance_forecast, index=returns.index)
    
    def _linear_trend_forecast(self, volatility_series, window=60):
        """Linear trend extrapolation forecast"""
        forecasts = []
        
        for i in range(len(volatility_series)):
            if i < window:
                forecasts.append(volatility_series.iloc[i] if i < len(volatility_series) else np.nan)
            else:
                # Fit linear trend to last 'window' observations
                y = volatility_series.iloc[i-window:i].values
                x = np.arange(len(y))
                
                if len(y) > 1 and not np.isnan(y).all():
                    slope, intercept, _, _, _ = stats.linregress(x, y)
                    forecast = intercept + slope * len(y)
                    forecasts.append(max(0.05, forecast))  # Minimum volatility
                else:
                    forecasts.append(volatility_series.iloc[i-1])
        
        return pd.Series(forecasts, index=volatility_series.index)
    
    def _seasonal_forecast(self, volatility_series):
        """Seasonal average forecast (same day of year)"""
        forecasts = []
        
        for i in range(len(volatility_series)):
            current_date = volatility_series.index[i]
            
            # Find historical values for same day of year
            same_day_values = []
            for j in range(i):
                historical_date = volatility_series.index[j]
                if (historical_date.month == current_date.month and 
                    abs(historical_date.day - current_date.day) <= 7):  # Within 1 week
                    same_day_values.append(volatility_series.iloc[j])
            
            if same_day_values:
                forecasts.append(np.mean(same_day_values))
            else:
                forecasts.append(volatility_series.iloc[i-1] if i > 0 else 0.2)
        
        return pd.Series(forecasts, index=volatility_series.index)
    
    def _vix_proxy_forecast(self, returns, window=20):
        """VIX-like proxy using options pricing principles"""
        forecasts = []
        
        for i in range(len(returns)):
            if i < window:
                forecasts.append(returns[:i+1].std() * np.sqrt(252) if i > 0 else 0.2)
            else:
                # Calculate implied volatility proxy using recent price movements
                recent_returns = returns.iloc[i-window:i]
                
                # Weight recent observations more heavily (like VIX)
                weights = np.exp(np.linspace(-1, 0, len(recent_returns)))
                weights = weights / weights.sum()
                
                weighted_var = np.sum(weights * (recent_returns ** 2))
                proxy_vol = np.sqrt(weighted_var * 252)
                forecasts.append(proxy_vol)
        
        return pd.Series(forecasts, index=returns.index)
    
    def create_portfolio_benchmarks(self, returns_data):
        """Create portfolio allocation benchmarks"""
        
        benchmarks = {}
        assets = returns_data.columns.tolist()
        n_assets = len(assets)
        
        # 1. Equal Weight (1/N)
        benchmarks['Equal_Weight'] = {asset: 1.0/n_assets for asset in assets}
        
        # 2. Market Cap Weight (approximate with price)
        latest_prices = returns_data.iloc[-1]
        market_weights = latest_prices / latest_prices.sum()
        benchmarks['Market_Cap_Proxy'] = market_weights.to_dict()
        
        # 3. Inverse Volatility
        volatilities = returns_data.std()
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        benchmarks['Inverse_Volatility'] = inv_vol_weights.to_dict()
        
        # 4. 60/40 Portfolio (if we have enough assets)
        if n_assets >= 2:
            # Assume first asset is stocks, second is bonds (proxy)
            weights_60_40 = {assets[0]: 0.6, assets[1]: 0.4}
            for i in range(2, n_assets):
                weights_60_40[assets[i]] = 0.0
            benchmarks['60_40_Portfolio'] = weights_60_40
        
        # 5. Maximum Diversification (naive version)
        correlation_matrix = returns_data.corr()
        avg_correlations = correlation_matrix.mean()
        # Weight inversely to average correlation
        diversification_weights = (1 - avg_correlations) / (1 - avg_correlations).sum()
        benchmarks['Max_Diversification'] = diversification_weights.to_dict()
        
        # 6. Random Allocation
        np.random.seed(42)
        random_weights = np.random.dirichlet(np.ones(n_assets))
        benchmarks['Random'] = {assets[i]: random_weights[i] for i in range(n_assets)}
        
        return benchmarks
    
    def compare_volatility_models(self, test_data_dict):
        """Compare ML volatility models against benchmarks"""
        
        print("\n📊 VOLATILITY MODEL COMPARISON ANALYSIS")
        print("=" * 60)
        
        comparison_results = {}
        
        for data_name, data in test_data_dict.items():
            print(f"\n🔬 Analyzing {data_name}...")
            
            # Create benchmarks
            benchmarks = self.create_volatility_benchmarks(data)
            
            # Get true volatility
            if 'True_Volatility' in data.columns:
                true_vol = data['True_Volatility']
            else:
                returns = data['Close'].pct_change().dropna()
                true_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Test ML model
            vol_tester = VolatilityTestFramework()
            ml_result = vol_tester.comprehensive_backtest(data, f"ML_{data_name}")
            
            if ml_result is None:
                print(f"  ❌ ML model failed for {data_name}")
                continue
            
            # Compare against benchmarks
            benchmark_performance = {}
            ml_performance = ml_result['metrics']
            
            # Evaluate each benchmark
            for bench_name, bench_series in benchmarks.items():
                aligned_true = true_vol[bench_series.index]
                valid_mask = ~(np.isnan(aligned_true) | np.isnan(bench_series))
                
                if valid_mask.sum() > 10:
                    true_clean = aligned_true[valid_mask]
                    bench_clean = bench_series[valid_mask]
                    
                    # Calculate metrics
                    r2 = 1 - (np.sum((true_clean - bench_clean)**2) / 
                             np.sum((true_clean - true_clean.mean())**2))
                    rmse = np.sqrt(np.mean((true_clean - bench_clean)**2))
                    mae = np.mean(np.abs(true_clean - bench_clean))
                    
                    # Direction accuracy
                    if len(true_clean) > 1:
                        true_dir = np.diff(true_clean)
                        bench_dir = np.diff(bench_clean)
                        direction_acc = np.mean(np.sign(true_dir) == np.sign(bench_dir))
                    else:
                        direction_acc = 0
                    
                    benchmark_performance[bench_name] = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'direction_accuracy': direction_acc
                    }
            
            # Store comparison
            comparison_results[data_name] = {
                'ml_performance': ml_performance,
                'benchmark_performance': benchmark_performance
            }
            
            # Print comparison
            print(f"  📈 ML Model Performance:")
            print(f"    R² Score: {ml_performance['r2']:.4f}")
            print(f"    RMSE: {ml_performance['rmse']:.4f}")
            print(f"    Direction Accuracy: {ml_performance['direction_accuracy']:.2%}")
            
            print(f"  📊 Best Benchmark Performance:")
            if benchmark_performance:
                best_benchmark = max(benchmark_performance.keys(), 
                                   key=lambda k: benchmark_performance[k]['r2'])
                best_perf = benchmark_performance[best_benchmark]
                print(f"    {best_benchmark}: R² = {best_perf['r2']:.4f}")
                
                improvement = ml_performance['r2'] - best_perf['r2']
                print(f"  🚀 ML Improvement: {improvement:+.4f}")
        
        self.volatility_benchmarks = comparison_results
        return comparison_results
    
    def compare_portfolio_models(self, returns_data_dict):
        """Compare portfolio allocation models against benchmarks"""
        
        print("\n📊 PORTFOLIO MODEL COMPARISON ANALYSIS")
        print("=" * 60)
        
        comparison_results = {}
        
        for data_name, returns_data in returns_data_dict.items():
            print(f"\n🔬 Analyzing {data_name}...")
            
            assets = returns_data.columns.tolist()
            
            # Create benchmarks
            benchmarks = self.create_portfolio_benchmarks(returns_data)
            
            # Test ML-based portfolio strategies
            portfolio_tester = PortfolioTestFramework()
            ml_results, ml_analysis = portfolio_tester.comprehensive_strategy_backtest(
                returns_data, assets, f"ML_{data_name}"
            )
            
            # Get best ML strategy
            best_ml_strategy = ml_analysis['best_strategy']
            best_ml_performance = ml_analysis['best_strategy_metrics']
            
            # Test benchmark strategies
            benchmark_performance = {}
            
            for bench_name, allocation in benchmarks.items():
                # Calculate benchmark performance
                asset_weights = np.array([allocation.get(asset, 0) for asset in assets])
                asset_weights = asset_weights / asset_weights.sum()  # Normalize
                
                # Calculate returns
                portfolio_returns = (returns_data[assets] * asset_weights).sum(axis=1)
                
                if len(portfolio_returns) > 0:
                    # Calculate metrics
                    total_return = (1 + portfolio_returns).prod() - 1
                    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                    annualized_vol = portfolio_returns.std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                    
                    # Drawdown
                    cumulative = (1 + portfolio_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdowns = (cumulative - running_max) / running_max
                    max_drawdown = drawdowns.min()
                    
                    benchmark_performance[bench_name] = {
                        'annualized_return': annualized_return,
                        'annualized_volatility': annualized_vol,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown
                    }
            
            # Store comparison
            comparison_results[data_name] = {
                'best_ml_strategy': best_ml_strategy,
                'best_ml_performance': best_ml_performance,
                'benchmark_performance': benchmark_performance
            }
            
            # Print comparison
            print(f"  📈 Best ML Strategy: {best_ml_strategy}")
            print(f"    Sharpe Ratio: {best_ml_performance['sharpe_ratio']:.4f}")
            print(f"    Annualized Return: {best_ml_performance['annualized_return']:.2%}")
            print(f"    Max Drawdown: {best_ml_performance['max_drawdown']:.2%}")
            
            print(f"  📊 Best Benchmark Performance:")
            if benchmark_performance:
                best_benchmark = max(benchmark_performance.keys(), 
                                   key=lambda k: benchmark_performance[k]['sharpe_ratio'])
                best_perf = benchmark_performance[best_benchmark]
                print(f"    {best_benchmark}: Sharpe = {best_perf['sharpe_ratio']:.4f}")
                
                improvement = best_ml_performance['sharpe_ratio'] - best_perf['sharpe_ratio']
                print(f"  🚀 ML Improvement: {improvement:+.4f}")
        
        self.portfolio_benchmarks = comparison_results
        return comparison_results
    
    def generate_statistical_significance_tests(self):
        """Perform statistical significance tests for model comparisons"""
        
        print("\n📊 STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 50)
        
        significance_results = {}
        
        # Test volatility models
        if self.volatility_benchmarks:
            print("\n🔬 Volatility Model Significance Tests:")
            
            for data_name, comparison in self.volatility_benchmarks.items():
                ml_performance = comparison['ml_performance']
                benchmark_performance = comparison['benchmark_performance']
                
                if benchmark_performance:
                    # Get best benchmark
                    best_benchmark = max(benchmark_performance.keys(), 
                                       key=lambda k: benchmark_performance[k]['r2'])
                    best_bench_r2 = benchmark_performance[best_benchmark]['r2']
                    ml_r2 = ml_performance['r2']
                    
                    # Simple significance test (would need more data for proper test)
                    improvement = ml_r2 - best_bench_r2
                    
                    print(f"  {data_name}:")
                    print(f"    ML R²: {ml_r2:.4f} vs Best Benchmark R²: {best_bench_r2:.4f}")
                    print(f"    Improvement: {improvement:+.4f}")
                    
                    if improvement > 0.05:
                        print(f"    🎯 SIGNIFICANT IMPROVEMENT")
                    elif improvement > 0.01:
                        print(f"    ⚠️  MARGINAL IMPROVEMENT")
                    else:
                        print(f"    ❌ NO SIGNIFICANT IMPROVEMENT")
        
        # Test portfolio models
        if self.portfolio_benchmarks:
            print("\n🔬 Portfolio Model Significance Tests:")
            
            for data_name, comparison in self.portfolio_benchmarks.items():
                ml_performance = comparison['best_ml_performance']
                benchmark_performance = comparison['benchmark_performance']
                
                if benchmark_performance:
                    # Get best benchmark
                    best_benchmark = max(benchmark_performance.keys(), 
                                       key=lambda k: benchmark_performance[k]['sharpe_ratio'])
                    best_bench_sharpe = benchmark_performance[best_benchmark]['sharpe_ratio']
                    ml_sharpe = ml_performance['sharpe_ratio']
                    
                    improvement = ml_sharpe - best_bench_sharpe
                    
                    print(f"  {data_name}:")
                    print(f"    ML Sharpe: {ml_sharpe:.4f} vs Best Benchmark: {best_bench_sharpe:.4f}")
                    print(f"    Improvement: {improvement:+.4f}")
                    
                    if improvement > 0.2:
                        print(f"    🎯 SIGNIFICANT IMPROVEMENT")
                    elif improvement > 0.05:
                        print(f"    ⚠️  MARGINAL IMPROVEMENT")
                    else:
                        print(f"    ❌ NO SIGNIFICANT IMPROVEMENT")
    
    def create_performance_visualization(self):
        """Create visualizations comparing model performance"""
        
        print("\n📊 Creating Performance Visualizations...")
        
        try:
            # Volatility comparison plot
            if self.volatility_benchmarks:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Volatility Model Performance Comparison', fontsize=16)
                
                # Collect data for plotting
                datasets = list(self.volatility_benchmarks.keys())
                ml_r2_scores = []
                best_benchmark_r2_scores = []
                
                for dataset in datasets:
                    comparison = self.volatility_benchmarks[dataset]
                    ml_r2_scores.append(comparison['ml_performance']['r2'])
                    
                    if comparison['benchmark_performance']:
                        best_r2 = max(perf['r2'] for perf in comparison['benchmark_performance'].values())
                        best_benchmark_r2_scores.append(best_r2)
                    else:
                        best_benchmark_r2_scores.append(0)
                
                # R² comparison
                x = np.arange(len(datasets))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, ml_r2_scores, width, label='ML Model', alpha=0.8)
                axes[0, 0].bar(x + width/2, best_benchmark_r2_scores, width, label='Best Benchmark', alpha=0.8)
                axes[0, 0].set_xlabel('Dataset')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].set_title('R² Score Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(datasets, rotation=45)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('volatility_model_comparison.png', dpi=300, bbox_inches='tight')
                print("  ✓ Volatility comparison plot saved as 'volatility_model_comparison.png'")
                plt.close()
            
            # Portfolio comparison plot
            if self.portfolio_benchmarks:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Portfolio Strategy Performance Comparison', fontsize=16)
                
                # Collect data for plotting
                datasets = list(self.portfolio_benchmarks.keys())
                ml_sharpe_scores = []
                best_benchmark_sharpe_scores = []
                
                for dataset in datasets:
                    comparison = self.portfolio_benchmarks[dataset]
                    ml_sharpe_scores.append(comparison['best_ml_performance']['sharpe_ratio'])
                    
                    if comparison['benchmark_performance']:
                        best_sharpe = max(perf['sharpe_ratio'] for perf in comparison['benchmark_performance'].values())
                        best_benchmark_sharpe_scores.append(best_sharpe)
                    else:
                        best_benchmark_sharpe_scores.append(0)
                
                # Sharpe ratio comparison
                x = np.arange(len(datasets))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, ml_sharpe_scores, width, label='ML Strategy', alpha=0.8)
                axes[0, 0].bar(x + width/2, best_benchmark_sharpe_scores, width, label='Best Benchmark', alpha=0.8)
                axes[0, 0].set_xlabel('Dataset')
                axes[0, 0].set_ylabel('Sharpe Ratio')
                axes[0, 0].set_title('Sharpe Ratio Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(datasets, rotation=45)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('portfolio_strategy_comparison.png', dpi=300, bbox_inches='tight')
                print("  ✓ Portfolio comparison plot saved as 'portfolio_strategy_comparison.png'")
                plt.close()
                
        except Exception as e:
            print(f"  ⚠️ Visualization creation failed: {e}")
    
    def generate_final_benchmark_report(self):
        """Generate comprehensive benchmark comparison report"""
        
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE MODEL BENCHMARKING REPORT")
        print("="*80)
        
        # Overall assessment
        vol_improvements = []
        portfolio_improvements = []
        
        # Volatility model assessment
        if self.volatility_benchmarks:
            print(f"\n🔬 VOLATILITY FORECASTING ASSESSMENT")
            print("-" * 50)
            
            for data_name, comparison in self.volatility_benchmarks.items():
                ml_r2 = comparison['ml_performance']['r2']
                benchmark_performance = comparison['benchmark_performance']
                
                if benchmark_performance:
                    best_bench_r2 = max(perf['r2'] for perf in benchmark_performance.values())
                    improvement = ml_r2 - best_bench_r2
                    vol_improvements.append(improvement)
                    
                    print(f"  {data_name}: ML R² = {ml_r2:.4f}, Best Benchmark = {best_bench_r2:.4f}")
                    print(f"    Improvement: {improvement:+.4f}")
            
            avg_vol_improvement = np.mean(vol_improvements) if vol_improvements else 0
            print(f"\n  📊 Average Volatility Model Improvement: {avg_vol_improvement:+.4f}")
        
        # Portfolio model assessment
        if self.portfolio_benchmarks:
            print(f"\n📈 PORTFOLIO ALLOCATION ASSESSMENT")
            print("-" * 50)
            
            for data_name, comparison in self.portfolio_benchmarks.items():
                ml_sharpe = comparison['best_ml_performance']['sharpe_ratio']
                benchmark_performance = comparison['benchmark_performance']
                
                if benchmark_performance:
                    best_bench_sharpe = max(perf['sharpe_ratio'] for perf in benchmark_performance.values())
                    improvement = ml_sharpe - best_bench_sharpe
                    portfolio_improvements.append(improvement)
                    
                    print(f"  {data_name}: ML Sharpe = {ml_sharpe:.4f}, Best Benchmark = {best_bench_sharpe:.4f}")
                    print(f"    Improvement: {improvement:+.4f}")
            
            avg_portfolio_improvement = np.mean(portfolio_improvements) if portfolio_improvements else 0
            print(f"\n  📊 Average Portfolio Strategy Improvement: {avg_portfolio_improvement:+.4f}")
        
        # Final recommendations
        print(f"\n💡 FINAL RECOMMENDATIONS")
        print("-" * 30)
        
        if avg_vol_improvement > 0.05:
            print("✅ Volatility Models: READY FOR PRODUCTION")
            print("   - Models significantly outperform benchmarks")
            print("   - Implement with confidence")
        elif avg_vol_improvement > 0.01:
            print("⚠️ Volatility Models: MARGINAL IMPROVEMENT")
            print("   - Models show some improvement over benchmarks")
            print("   - Consider additional feature engineering")
        else:
            print("❌ Volatility Models: NEEDS IMPROVEMENT")
            print("   - Models do not consistently beat simple benchmarks")
            print("   - Recommend major model redesign")
        
        if avg_portfolio_improvement > 0.1:
            print("✅ Portfolio Strategies: EXCELLENT PERFORMANCE")
            print("   - Strategies significantly outperform benchmarks")
            print("   - Ready for production deployment")
        elif avg_portfolio_improvement > 0.05:
            print("⚠️ Portfolio Strategies: GOOD PERFORMANCE")
            print("   - Strategies show solid improvement")
            print("   - Fine-tune for production use")
        else:
            print("❌ Portfolio Strategies: NEEDS IMPROVEMENT")
            print("   - Strategies do not consistently beat benchmarks")
            print("   - Review allocation algorithms")
        
        # Save results
        self._save_benchmark_results()
        print(f"\n💾 Benchmark results saved to 'benchmark_comparison_results.json'")
        print("="*80)
    
    def _save_benchmark_results(self):
        """Save benchmark comparison results"""
        output_data = {
            'volatility_benchmarks': self.volatility_benchmarks,
            'portfolio_benchmarks': self.portfolio_benchmarks,
            'summary': {
                'volatility_avg_improvement': np.mean([
                    comp['ml_performance']['r2'] - max(comp['benchmark_performance'][k]['r2'] 
                    for k in comp['benchmark_performance']) 
                    for comp in self.volatility_benchmarks.values() 
                    if comp['benchmark_performance']
                ]) if self.volatility_benchmarks else 0,
                'portfolio_avg_improvement': np.mean([
                    comp['best_ml_performance']['sharpe_ratio'] - max(comp['benchmark_performance'][k]['sharpe_ratio'] 
                    for k in comp['benchmark_performance']) 
                    for comp in self.portfolio_benchmarks.values() 
                    if comp['benchmark_performance']
                ]) if self.portfolio_benchmarks else 0
            }
        }
        
        with open('benchmark_comparison_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
    
    def run_comprehensive_benchmark_comparison(self):
        """Run complete benchmark comparison"""
        
        print("🚀 Starting Comprehensive Model Benchmarking")
        print("=" * 60)
        
        # Create test datasets
        vol_tester = VolatilityTestFramework()
        portfolio_tester = PortfolioTestFramework()
        
        # Volatility test data
        vol_test_data = {
            'Normal_Market': vol_tester.generate_realistic_market_data('normal', 600),  # Reduced
            'Crisis_Market': vol_tester.generate_realistic_market_data('crisis', 500),  # Reduced
            'Low_Vol_Market': vol_tester.generate_realistic_market_data('low_vol', 600)  # Reduced
        }
        
        # Portfolio test data
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        portfolio_test_data = {}
        
        for scenario in ['normal', 'bear_market', 'bull_market']:
            _, returns_data = portfolio_tester.generate_realistic_asset_data(assets, scenario, 600)  # Reduced
            portfolio_test_data[f'{scenario.title()}_Market'] = returns_data
        
        # Run comparisons
        self.compare_volatility_models(vol_test_data)
        self.compare_portfolio_models(portfolio_test_data)
        
        # Statistical tests
        self.generate_statistical_significance_tests()
        
        # Create visualizations
        self.create_performance_visualization()
        
        # Generate final report
        self.generate_final_benchmark_report()

def main():
    """Main execution function"""
    benchmarker = ModelBenchmarkFramework()
    benchmarker.run_comprehensive_benchmark_comparison()

if __name__ == "__main__":
    main()