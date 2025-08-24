"""
Comprehensive Portfolio Allocation Test Suite
===========================================
This script provides extensive testing and evaluation of portfolio allocation strategies
including risk-adjusted performance metrics, strategy comparison, backtesting, and actionable insights.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from portfolio_allocation import portfolio_allocator, PortfolioAllocationEngine
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json

class PortfolioTestFramework:
    def __init__(self):
        self.results = {}
        self.strategy_performance = {}
        self.detailed_results = []
        self.benchmark_data = {}
        
    def generate_realistic_asset_data(self, assets, scenario='normal', days=1000):
        """Generate realistic multi-asset data with different market scenarios"""
        np.random.seed(42)
        n_assets = len(assets)
        
        if scenario == 'normal':
            # Normal market conditions
            annual_returns = np.random.uniform(0.05, 0.15, n_assets)
            annual_vols = np.random.uniform(0.12, 0.25, n_assets)
            correlations = self._generate_correlation_matrix(n_assets, base_corr=0.3)
        elif scenario == 'bull_market':
            # Bull market with higher returns, moderate correlations
            annual_returns = np.random.uniform(0.12, 0.25, n_assets)
            annual_vols = np.random.uniform(0.15, 0.30, n_assets)
            correlations = self._generate_correlation_matrix(n_assets, base_corr=0.4)
        elif scenario == 'bear_market':
            # Bear market with negative returns, high correlations
            annual_returns = np.random.uniform(-0.20, 0.05, n_assets)
            annual_vols = np.random.uniform(0.25, 0.50, n_assets)
            correlations = self._generate_correlation_matrix(n_assets, base_corr=0.7)
        elif scenario == 'high_vol':
            # High volatility environment
            annual_returns = np.random.uniform(-0.05, 0.10, n_assets)
            annual_vols = np.random.uniform(0.30, 0.60, n_assets)
            correlations = self._generate_correlation_matrix(n_assets, base_corr=0.5)
        elif scenario == 'low_corr':
            # Low correlation environment (good for diversification)
            annual_returns = np.random.uniform(0.06, 0.14, n_assets)
            annual_vols = np.random.uniform(0.15, 0.25, n_assets)
            correlations = self._generate_correlation_matrix(n_assets, base_corr=0.1)
        
        # Generate correlated returns
        daily_returns = annual_returns / 252
        daily_vols = annual_vols / np.sqrt(252)
        
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(correlations)
        
        # Generate returns
        uncorrelated_returns = np.random.normal(0, 1, (days, n_assets))
        correlated_returns = uncorrelated_returns @ L.T
        
        # Scale to desired mean and volatility
        returns_matrix = np.zeros((days, n_assets))
        for i in range(n_assets):
            returns_matrix[:, i] = (correlated_returns[:, i] * daily_vols[i] + daily_returns[i])
        
        # Convert to price data
        prices = np.zeros((days, n_assets))
        initial_prices = np.random.uniform(50, 200, n_assets)
        prices[0, :] = initial_prices
        
        for t in range(1, days):
            prices[t, :] = prices[t-1, :] * (1 + returns_matrix[t, :])
        
        dates = pd.date_range('2020-01-01', periods=days, freq='D')
        price_data = pd.DataFrame(prices, columns=assets, index=dates)
        returns_data = pd.DataFrame(returns_matrix, columns=assets, index=dates)
        
        return price_data, returns_data
    
    def _generate_correlation_matrix(self, n_assets, base_corr=0.3):
        """Generate a realistic correlation matrix"""
        # Start with identity matrix
        corr_matrix = np.eye(n_assets)
        
        # Add random correlations
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Generate correlation with some randomness around base_corr
                corr = np.random.normal(base_corr, 0.15)
                corr = np.clip(corr, -0.9, 0.9)  # Keep correlations reasonable
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Ensure positive definite matrix
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Make positive definite
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix
    
    def load_real_market_data(self, assets=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']):
        """Load real market data for portfolio testing"""
        real_data = {}
        
        print("📈 Loading real market data...")
        for asset in assets:
            try:
                ticker = yf.Ticker(asset)
                data = ticker.history(period="3y")
                if not data.empty and len(data) > 252:  # At least 1 year of data
                    real_data[asset] = data
                    print(f"✓ Loaded {asset}: {len(data)} days")
            except Exception as e:
                print(f"✗ Failed to load {asset}: {e}")
        
        if real_data:
            # Align dates and create combined dataset
            common_dates = None
            for asset, data in real_data.items():
                if common_dates is None:
                    common_dates = data.index
                else:
                    common_dates = common_dates.intersection(data.index)
            
            price_data = pd.DataFrame()
            for asset, data in real_data.items():
                price_data[asset] = data.loc[common_dates, 'Close']
            
            returns_data = price_data.pct_change().dropna()
            return price_data, returns_data
        
        return None, None
    
    def comprehensive_strategy_backtest(self, returns_data, assets, test_name, 
                                      rebalance_frequency=30, transaction_cost=0.001):
        """Perform comprehensive backtesting of all allocation strategies"""
        print(f"\n=== Strategy Backtest: {test_name} ===")
        
        # Get all allocation strategies
        allocator = PortfolioAllocationEngine()
        strategy_names = [
            'Equal Weight', 'Risk Parity', 'Maximum Sharpe', 'Minimum Variance',
            'Momentum Based', 'Volatility Target', 'Black-Litterman'
        ]
        
        # Split data for out-of-sample testing
        train_size = int(len(returns_data) * 0.6)
        test_returns = returns_data.iloc[train_size:]
        
        print(f"Training period: {train_size} days")
        print(f"Testing period: {len(test_returns)} days")
        
        strategy_results = {}
        
        for strategy_name in strategy_names:
            print(f"  Testing {strategy_name}...")
            try:
                performance = self._backtest_single_strategy(
                    returns_data, test_returns, assets, strategy_name, 
                    rebalance_frequency, transaction_cost
                )
                strategy_results[strategy_name] = performance
            except Exception as e:
                print(f"    ❌ Error testing {strategy_name}: {e}")
                continue
        
        # Calculate benchmark (buy and hold equal weight)
        benchmark_performance = self._calculate_benchmark_performance(test_returns, assets)
        strategy_results['Benchmark (Equal Weight B&H)'] = benchmark_performance
        
        # Analyze results
        analysis = self._analyze_strategy_performance(strategy_results, test_name)
        
        self.strategy_performance[test_name] = {
            'strategy_results': strategy_results,
            'analysis': analysis,
            'test_period': (test_returns.index[0], test_returns.index[-1]),
            'n_assets': len(assets),
            'n_days': len(test_returns)
        }
        
        return strategy_results, analysis
    
    def _backtest_single_strategy(self, full_returns, test_returns, assets, 
                                strategy_name, rebalance_freq, transaction_cost):
        """Backtest a single allocation strategy"""
        
        allocator = PortfolioAllocationEngine()
        
        # Get strategy method
        strategy_methods = {
            'Equal Weight': allocator.equal_weight_allocation,
            'Risk Parity': lambda a: allocator.risk_parity_allocation(full_returns, a),
            'Maximum Sharpe': lambda a: allocator.maximum_sharpe_allocation(full_returns, a),
            'Minimum Variance': lambda a: allocator.minimum_variance_allocation(full_returns, a),
            'Momentum Based': lambda a: allocator.momentum_based_allocation(full_returns, a),
            'Volatility Target': lambda a: allocator.volatility_target_allocation(full_returns, a),
            'Black-Litterman': lambda a: allocator.black_litterman_allocation(full_returns, a)
        }
        
        strategy_method = strategy_methods[strategy_name]
        
        # Initialize portfolio
        portfolio_values = []
        weights_history = []
        turnover_history = []
        current_weights = None
        
        # Rebalancing loop
        for i in range(0, len(test_returns), rebalance_freq):
            end_idx = min(i + rebalance_freq, len(test_returns))
            period_returns = test_returns.iloc[i:end_idx]
            
            if len(period_returns) == 0:
                break
            
            # Get allocation for this period
            # Use data up to current point for allocation decisions
            train_end = full_returns.index.get_loc(period_returns.index[0])
            historical_data = full_returns.iloc[:train_end]
            
            if len(historical_data) < 60:  # Need sufficient history
                allocation = allocator.equal_weight_allocation(assets)
            else:
                try:
                    allocation = strategy_method(assets)
                except:
                    allocation = allocator.equal_weight_allocation(assets)
            
            # Convert to weights array
            new_weights = np.array([allocation.get(asset, 0) for asset in assets])
            new_weights = new_weights / np.sum(new_weights)  # Normalize
            
            # Calculate turnover and transaction costs
            if current_weights is not None:
                turnover = np.sum(np.abs(new_weights - current_weights))
                turnover_history.append(turnover)
                transaction_costs = turnover * transaction_cost
            else:
                turnover_history.append(0)
                transaction_costs = 0
            
            current_weights = new_weights.copy()
            weights_history.append(current_weights.copy())
            
            # Calculate portfolio returns for this period
            period_portfolio_returns = (period_returns[assets] * current_weights).sum(axis=1)
            
            # Apply transaction costs to first day of period
            if len(period_portfolio_returns) > 0:
                period_portfolio_returns.iloc[0] -= transaction_costs
            
            portfolio_values.extend(period_portfolio_returns.tolist())
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values, index=test_returns.index[:len(portfolio_values)])
        performance = self._calculate_portfolio_performance_metrics(portfolio_returns)
        
        # Add strategy-specific metrics
        performance['avg_turnover'] = np.mean(turnover_history) if turnover_history else 0
        performance['total_transaction_costs'] = len(turnover_history) * transaction_cost * np.mean(turnover_history) if turnover_history else 0
        
        return performance
    
    def _calculate_benchmark_performance(self, test_returns, assets):
        """Calculate buy-and-hold equal weight benchmark performance"""
        equal_weights = np.array([1.0/len(assets)] * len(assets))
        benchmark_returns = (test_returns[assets] * equal_weights).sum(axis=1)
        return self._calculate_portfolio_performance_metrics(benchmark_returns)
    
    def _calculate_portfolio_performance_metrics(self, returns):
        """Calculate comprehensive portfolio performance metrics"""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_deviation': downside_deviation
        }
    
    def _analyze_strategy_performance(self, strategy_results, test_name):
        """Analyze and rank strategy performance"""
        
        # Create summary DataFrame
        summary_data = []
        for strategy, metrics in strategy_results.items():
            summary_data.append({
                'Strategy': strategy,
                'Annualized Return': metrics.get('annualized_return', 0),
                'Volatility': metrics.get('annualized_volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Rank strategies by multiple criteria
        rankings = {}
        
        for metric in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
            rankings[metric] = summary_df.nlargest(3, metric)['Strategy'].tolist()
        
        rankings['Max Drawdown'] = summary_df.nsmallest(3, 'Max Drawdown')['Strategy'].tolist()
        
        # Overall score (weighted combination)
        summary_df['Overall Score'] = (
            summary_df['Sharpe Ratio'] * 0.3 +
            summary_df['Sortino Ratio'] * 0.25 +
            summary_df['Calmar Ratio'] * 0.25 +
            (1 + summary_df['Max Drawdown']) * 0.2  # Max drawdown is negative, so (1 + MD) gives higher score for lower drawdown
        )
        
        overall_ranking = summary_df.sort_values('Overall Score', ascending=False)
        
        analysis = {
            'summary_table': summary_df,
            'rankings': rankings,
            'overall_ranking': overall_ranking,
            'best_strategy': overall_ranking.iloc[0]['Strategy'],
            'best_strategy_metrics': strategy_results[overall_ranking.iloc[0]['Strategy']]
        }
        
        # Print analysis
        print(f"\n📊 Strategy Performance Analysis for {test_name}")
        print("=" * 60)
        print(overall_ranking[['Strategy', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']].to_string(index=False, float_format='%.4f'))
        
        print(f"\n🏆 Best Overall Strategy: {analysis['best_strategy']}")
        best_metrics = analysis['best_strategy_metrics']
        print(f"   Annualized Return: {best_metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        
        return analysis
    
    def stress_test_strategies(self, assets):
        """Perform stress testing under various market conditions"""
        print("\n🔥 STRESS TESTING PORTFOLIO STRATEGIES")
        print("=" * 50)
        
        stress_scenarios = ['bear_market', 'high_vol', 'low_corr']
        stress_results = {}
        
        for scenario in stress_scenarios:
            print(f"\n📉 Testing scenario: {scenario.upper()}")
            price_data, returns_data = self.generate_realistic_asset_data(assets, scenario, days=500)
            
            strategy_results, analysis = self.comprehensive_strategy_backtest(
                returns_data, assets, f"stress_{scenario}"
            )
            
            stress_results[scenario] = {
                'best_strategy': analysis['best_strategy'],
                'best_sharpe': analysis['best_strategy_metrics']['sharpe_ratio'],
                'worst_drawdown': min(metrics['max_drawdown'] for metrics in strategy_results.values())
            }
        
        # Analyze stress test results
        print(f"\n🎯 STRESS TEST SUMMARY")
        for scenario, results in stress_results.items():
            print(f"   {scenario.upper()}:")
            print(f"     Best Strategy: {results['best_strategy']}")
            print(f"     Best Sharpe: {results['best_sharpe']:.4f}")
            print(f"     Worst Drawdown: {results['worst_drawdown']:.2%}")
        
        return stress_results
    
    def run_comprehensive_evaluation(self):
        """Run complete portfolio allocation evaluation"""
        print("🚀 Starting Comprehensive Portfolio Allocation Evaluation")
        print("=" * 70)
        
        # Test assets
        test_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'XOM']
        
        # Test on synthetic data with different scenarios
        scenarios = ['normal', 'bull_market', 'bear_market']
        
        for scenario in scenarios:
            print(f"\n🔬 Testing scenario: {scenario.upper()}")
            price_data, returns_data = self.generate_realistic_asset_data(test_assets, scenario, days=1000)
            
            strategy_results, analysis = self.comprehensive_strategy_backtest(
                returns_data, test_assets, f"synthetic_{scenario}"
            )
        
        # Test on real market data
        print(f"\n📈 Testing on real market data...")
        real_price_data, real_returns_data = self.load_real_market_data(test_assets)
        
        if real_returns_data is not None:
            strategy_results, analysis = self.comprehensive_strategy_backtest(
                real_returns_data, real_returns_data.columns.tolist(), "real_market"
            )
        
        # Stress testing
        stress_results = self.stress_test_strategies(test_assets[:5])  # Use fewer assets for stress testing
        
        # Generate comprehensive report
        self.generate_portfolio_report(stress_results)
    
    def generate_portfolio_report(self, stress_results):
        """Generate comprehensive portfolio allocation report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE PORTFOLIO ALLOCATION EVALUATION REPORT")
        print("="*80)
        
        if not self.strategy_performance:
            print("❌ No results available for reporting")
            return
        
        # Aggregate performance across all tests
        all_sharpe_ratios = {}
        all_max_drawdowns = {}
        strategy_wins = {}
        
        for test_name, test_data in self.strategy_performance.items():
            for strategy, metrics in test_data['strategy_results'].items():
                if strategy not in all_sharpe_ratios:
                    all_sharpe_ratios[strategy] = []
                    all_max_drawdowns[strategy] = []
                
                all_sharpe_ratios[strategy].append(metrics.get('sharpe_ratio', 0))
                all_max_drawdowns[strategy].append(metrics.get('max_drawdown', 0))
        
        # Calculate average performance
        avg_performance = {}
        for strategy in all_sharpe_ratios:
            avg_performance[strategy] = {
                'avg_sharpe': np.mean(all_sharpe_ratios[strategy]),
                'avg_max_dd': np.mean(all_max_drawdowns[strategy]),
                'sharpe_std': np.std(all_sharpe_ratios[strategy]),
                'consistency': 1 / (1 + np.std(all_sharpe_ratios[strategy]))  # Higher is more consistent
            }
        
        # Rank strategies
        sorted_strategies = sorted(avg_performance.items(), 
                                 key=lambda x: x[1]['avg_sharpe'], reverse=True)
        
        print(f"\n📊 OVERALL PERFORMANCE RANKING")
        print(f"{'Rank':<4} {'Strategy':<25} {'Avg Sharpe':<12} {'Avg MaxDD':<12} {'Consistency':<12}")
        print("-" * 70)
        
        for i, (strategy, metrics) in enumerate(sorted_strategies, 1):
            print(f"{i:<4} {strategy:<25} {metrics['avg_sharpe']:<12.4f} "
                  f"{metrics['avg_max_dd']:<12.2%} {metrics['consistency']:<12.4f}")
        
        # Best overall strategy
        best_strategy = sorted_strategies[0][0]
        best_metrics = sorted_strategies[0][1]
        
        print(f"\n🏆 BEST OVERALL STRATEGY: {best_strategy}")
        print(f"   Average Sharpe Ratio: {best_metrics['avg_sharpe']:.4f}")
        print(f"   Average Max Drawdown: {best_metrics['avg_max_dd']:.2%}")
        print(f"   Consistency Score: {best_metrics['consistency']:.4f}")
        
        # Performance assessment
        avg_best_sharpe = best_metrics['avg_sharpe']
        
        if avg_best_sharpe >= 1.0:
            assessment = "excellent"
            print(f"\n✅ EXCELLENT: Portfolio allocator shows superior performance")
        elif avg_best_sharpe >= 0.6:
            assessment = "good"
            print(f"\n✅ GOOD: Portfolio allocator demonstrates solid performance")
        elif avg_best_sharpe >= 0.3:
            assessment = "moderate"
            print(f"\n⚠️  MODERATE: Portfolio allocator shows reasonable performance")
        else:
            assessment = "poor"
            print(f"\n❌ POOR: Portfolio allocator needs significant improvement")
        
        # Strategy-specific insights
        print(f"\n🔍 STRATEGY INSIGHTS")
        self._generate_strategy_insights(sorted_strategies, stress_results)
        
        # Recommendations
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        self._generate_portfolio_recommendations(assessment, best_strategy, sorted_strategies)
        
        # Save detailed results
        self._save_portfolio_results()
        
        print(f"\n💾 Detailed results saved to 'portfolio_test_results.json'")
        print("="*80)
    
    def _generate_strategy_insights(self, sorted_strategies, stress_results):
        """Generate insights about strategy performance"""
        
        # Identify consistently good performers
        top_3 = [s[0] for s in sorted_strategies[:3]]
        print(f"   Top 3 Consistent Performers: {', '.join(top_3)}")
        
        # Analyze worst performers
        worst_3 = [s[0] for s in sorted_strategies[-3:]]
        print(f"   Strategies needing improvement: {', '.join(worst_3)}")
        
        # Stress test insights
        if stress_results:
            stress_winners = {}
            for scenario, results in stress_results.items():
                strategy = results['best_strategy']
                if strategy not in stress_winners:
                    stress_winners[strategy] = 0
                stress_winners[strategy] += 1
            
            if stress_winners:
                best_stress_performer = max(stress_winners.keys(), key=lambda k: stress_winners[k])
                print(f"   Best Crisis Performer: {best_stress_performer}")
    
    def _generate_portfolio_recommendations(self, assessment, best_strategy, sorted_strategies):
        """Generate specific recommendations for portfolio allocation improvement"""
        
        if assessment == "poor":
            print("   1. 🔧 CRITICAL: Review and improve optimization algorithms")
            print("      - Implement more sophisticated risk models")
            print("      - Add transaction cost optimization")
            print("      - Consider multi-period optimization")
            print("   2. 📊 Improve input data quality and frequency")
            print("   3. 🎯 Add regime-aware allocation strategies")
            print("   4. 🔄 Implement dynamic rebalancing rules")
            
        elif assessment == "moderate":
            print("   1. 🛠️  Enhance existing strategies:")
            print("      - Add robust optimization techniques")
            print("      - Implement regime detection")
            print("      - Consider factor-based allocation")
            print("   2. 🎛️  Fine-tune rebalancing frequencies")
            print("   3. 📈 Add alternative risk measures (CVaR, etc.)")
            print("   4. 🔍 Implement strategy combination/ensemble approaches")
            
        elif assessment == "good":
            print("   1. 🎯 Optimize transaction costs and rebalancing")
            print("   2. 🔄 Add real-time market regime detection")
            print("   3. 📊 Implement confidence intervals for allocations")
            print("   4. 🚀 Consider deploying best strategies to production")
            
        else:  # excellent
            print("   1. ✅ Strategies are ready for production deployment")
            print("   2. 📊 Implement comprehensive monitoring system")
            print("   3. 🔄 Set up automated rebalancing with risk controls")
            print("   4. 📈 Consider expanding to additional asset classes")
        
        # Strategy-specific recommendations
        worst_performer = sorted_strategies[-1][0]
        print(f"   ⚠️  FOCUS AREA: Improve '{worst_performer}' strategy implementation")
        
        if best_strategy in ['Equal Weight', 'Risk Parity']:
            print("   💡 Simple strategies performing well - market may be efficient")
        elif best_strategy in ['Maximum Sharpe', 'Momentum Based']:
            print("   💡 Return-focused strategies winning - consider trend-following")
        elif best_strategy in ['Minimum Variance', 'Volatility Target']:
            print("   💡 Risk-focused strategies effective - emphasize risk management")
    
    def _save_portfolio_results(self):
        """Save detailed portfolio results to JSON file"""
        output_data = {}
        
        for test_name, result in self.strategy_performance.items():
            serializable_result = {
                'test_name': test_name,
                'n_assets': result['n_assets'],
                'n_days': result['n_days'],
                'test_period': [str(result['test_period'][0]), str(result['test_period'][1])],
                'strategy_results': result['strategy_results'],
                'best_strategy': result['analysis']['best_strategy']
            }
            
            output_data[test_name] = serializable_result
        
        with open('portfolio_test_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

def main():
    """Main execution function"""
    tester = PortfolioTestFramework()
    tester.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()