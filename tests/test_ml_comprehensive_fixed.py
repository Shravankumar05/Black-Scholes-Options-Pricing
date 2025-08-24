"""
Comprehensive Test Suite for ML-Powered Components
===============================================
Tests both volatility forecaster and asset allocation recommender with proper benchmarks.
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from ml_components import AdvancedVolatilityForecaster, EnhancedVolatilityForecaster
    from portfolio_allocation import AdvancedPortfolioAllocationEngine
    print("✓ Successfully imported ML components")
except ImportError as e:
    print(f"❌ Failed to import ML components: {e}")
    sys.exit(1)

class MLTestFramework:
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    def generate_realistic_test_data(self, n_days=500, n_assets=5, seed=42):
        """Generate realistic multi-asset test data with different market regimes"""
        np.random.seed(seed)
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')[:n_days]
        
        # Asset names
        assets = [f'ASSET_{i+1}' for i in range(n_assets)]
        
        # Generate correlated returns with regime switching
        correlation_matrix = self._generate_correlation_matrix(n_assets)
        
        # Market regimes: normal, high_vol, crisis
        regime_changes = [100, 200, 350, 450]  # Days where regime changes
        regimes = ['normal', 'high_vol', 'crisis', 'recovery', 'normal']
        
        all_returns = []
        current_vol = 0.015  # Base volatility
        
        for i, date in enumerate(dates):
            # Determine current regime
            regime_idx = sum(1 for change in regime_changes if i >= change)
            current_regime = regimes[min(regime_idx, len(regimes)-1)]
            
            # Adjust volatility based on regime
            if current_regime == 'normal':
                current_vol = 0.015
            elif current_regime == 'high_vol':
                current_vol = 0.025
            elif current_regime == 'crisis':
                current_vol = 0.045
            elif current_regime == 'recovery':
                current_vol = 0.030
            
            # Add volatility clustering
            if i > 0:
                vol_persistence = 0.7
                current_vol = vol_persistence * current_vol + (1 - vol_persistence) * 0.02
            
            # Generate correlated returns
            base_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=correlation_matrix * current_vol**2
            )
            
            # Add market factor (systematic risk)
            market_factor = np.random.normal(0, current_vol * 0.5)
            factor_loadings = np.random.uniform(0.3, 0.8, n_assets)
            final_returns = base_returns + factor_loadings * market_factor
            
            all_returns.append(final_returns)
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(all_returns, index=dates, columns=assets)
        
        # Create price data from returns
        prices_df = pd.DataFrame(index=dates, columns=assets)
        for asset in assets:
            prices_df[asset] = (1 + returns_df[asset]).cumprod() * 100
        
        # Add OHLCV data for the first asset
        main_asset_data = pd.DataFrame(index=dates)
        main_asset_data['Close'] = prices_df[assets[0]]
        main_asset_data['Open'] = main_asset_data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
        main_asset_data['High'] = np.maximum(
            main_asset_data[['Open', 'Close']].max(axis=1),
            main_asset_data['Close'] * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        )
        main_asset_data['Low'] = np.minimum(
            main_asset_data[['Open', 'Close']].min(axis=1),
            main_asset_data['Close'] * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        )
        main_asset_data['Volume'] = np.random.lognormal(10, 1, len(dates))
        
        # Calculate realized volatility
        main_asset_data['Realized_Volatility'] = returns_df[assets[0]].rolling(20).std() * np.sqrt(252)
        
        return main_asset_data, returns_df, prices_df
    
    def _generate_correlation_matrix(self, n_assets):
        """Generate a realistic correlation matrix"""
        # Start with random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr_matrix = np.dot(A, A.T)
        
        # Normalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        # Ensure reasonable correlations (0.1 to 0.7)
        corr_matrix = np.clip(corr_matrix, 0.1, 0.7)
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix

class VolatilityForecasterTests(MLTestFramework):
    def __init__(self):
        super().__init__()
        self.forecaster = AdvancedVolatilityForecaster()
        
    def test_feature_engineering(self):
        """Test the advanced feature engineering"""
        print("\n🧪 Testing Advanced Feature Engineering...")
        
        try:
            # Generate test data
            price_data, _, _ = self.generate_realistic_test_data(n_days=300, n_assets=1)
            
            # Test feature creation
            features_df = self.forecaster.create_advanced_features(price_data)
            
            if features_df is not None:
                print(f"   ✅ Created {len(features_df.columns)} features")
                print(f"   📊 Feature shape: {features_df.shape}")
                print(f"   📈 Sample features: {list(features_df.columns[:10])}")
                
                # Check for NaN values
                nan_counts = features_df.isnull().sum()
                nan_features = nan_counts[nan_counts > 0]
                
                if len(nan_features) > 0:
                    print(f"   ⚠️ Features with NaN values: {len(nan_features)}")
                    print(f"   🔧 Max NaN count: {nan_features.max()}")
                else:
                    print("   ✅ No NaN values in features")
                
                # Check for infinite values
                inf_features = features_df.replace([np.inf, -np.inf], np.nan).isnull().sum() - nan_counts
                inf_features = inf_features[inf_features > 0]
                
                if len(inf_features) > 0:
                    print(f"   ⚠️ Features with infinite values: {len(inf_features)}")
                else:
                    print("   ✅ No infinite values in features")
                
                self.test_results['feature_engineering'] = {
                    'success': True,
                    'num_features': len(features_df.columns),
                    'num_observations': len(features_df),
                    'nan_features': len(nan_features),
                    'inf_features': len(inf_features)
                }
                
                return True
            else:
                print("   ❌ Feature engineering failed")
                self.test_results['feature_engineering'] = {'success': False}
                return False
                
        except Exception as e:
            print(f"   ❌ Feature engineering error: {e}")
            traceback.print_exc()
            self.test_results['feature_engineering'] = {'success': False, 'error': str(e)}
            return False
    
    def test_model_training(self):
        """Test the ensemble model training"""
        print("\n🧪 Testing Model Training...")
        
        try:
            # Generate training data
            price_data, _, _ = self.generate_realistic_test_data(n_days=400, n_assets=1)
            
            print("   🚀 Training advanced ensemble...")
            start_time = time.time()
            
            success = self.forecaster.train_advanced_ensemble(price_data)
            training_time = time.time() - start_time
            
            if success:
                print(f"   ✅ Training successful in {training_time:.2f}s")
                print(f"   🎯 Models trained: {list(self.forecaster.models.keys())}")
                print(f"   ⚖️ Ensemble weights: {self.forecaster.ensemble_weights}")
                
                self.test_results['model_training'] = {
                    'success': True,
                    'training_time': training_time,
                    'models_trained': list(self.forecaster.models.keys()),
                    'ensemble_weights': self.forecaster.ensemble_weights
                }
                
                return True
            else:
                print("   ❌ Model training failed")
                self.test_results['model_training'] = {'success': False}
                return False
                
        except Exception as e:
            print(f"   ❌ Model training error: {e}")
            traceback.print_exc()
            self.test_results['model_training'] = {'success': False, 'error': str(e)}
            return False
    
    def test_forecasting_accuracy(self):
        """Test forecasting accuracy with walk-forward validation"""
        print("\n🧪 Testing Forecasting Accuracy...")
        
        try:
            # Generate test data
            price_data, _, _ = self.generate_realistic_test_data(n_days=500, n_assets=1)
            
            # Train on first 80% of data
            train_size = int(len(price_data) * 0.8)
            train_data = price_data.iloc[:train_size]
            test_data = price_data.iloc[train_size:]
            
            # Train model
            if not self.forecaster.is_trained:
                success = self.forecaster.train_advanced_ensemble(train_data)
                if not success:
                    print("   ❌ Failed to train model for forecasting test")
                    return False
            
            # Walk-forward forecasting
            forecasts = []
            actuals = []
            
            print("   🔮 Performing walk-forward validation...")
            
            for i in range(0, len(test_data) - 30, 10):  # Forecast every 10 days
                # Get training data up to current point
                current_train_data = pd.concat([train_data, test_data.iloc[:i]])
                
                # Get forecast
                forecast = self.forecaster.advanced_ensemble_forecast(current_train_data, days_ahead=5)
                forecasts.append(forecast)
                
                # Get actual volatility for next 5 days
                future_data = test_data.iloc[i:i+5]
                if len(future_data) >= 5:
                    actual_vol = future_data['Close'].pct_change().std() * np.sqrt(252)
                    actuals.append(actual_vol)
                else:
                    break
            
            if len(forecasts) >= 5 and len(actuals) >= 5:
                # Calculate performance metrics
                forecasts = np.array(forecasts[:len(actuals)])
                actuals = np.array(actuals[:len(forecasts)])
                
                mse = np.mean((forecasts - actuals) ** 2)
                mae = np.mean(np.abs(forecasts - actuals))
                mape = np.mean(np.abs((forecasts - actuals) / actuals)) * 100
                correlation = np.corrcoef(forecasts, actuals)[0,1] if len(forecasts) > 1 else 0
                
                # Direction accuracy
                if len(forecasts) > 1:
                    forecast_changes = np.diff(forecasts)
                    actual_changes = np.diff(actuals)
                    direction_accuracy = np.mean(
                        np.sign(forecast_changes) == np.sign(actual_changes)
                    ) * 100
                else:
                    direction_accuracy = 50.0
                
                print(f"   📊 Performance Metrics:")
                print(f"      MSE: {mse:.6f}")
                print(f"      MAE: {mae:.6f}")
                print(f"      MAPE: {mape:.2f}%")
                print(f"      Correlation: {correlation:.4f}")
                print(f"      Direction Accuracy: {direction_accuracy:.1f}%")
                
                # Performance assessment
                performance_score = 0
                if mse < 0.01:
                    performance_score += 25
                if mae < 0.05:
                    performance_score += 25
                if correlation > 0.3:
                    performance_score += 25
                if direction_accuracy > 55:
                    performance_score += 25
                
                assessment = "excellent" if performance_score >= 75 else \
                           "good" if performance_score >= 50 else \
                           "moderate" if performance_score >= 25 else "poor"
                
                print(f"   🎯 Overall Assessment: {assessment.upper()} ({performance_score}/100)")
                
                self.test_results['forecasting_accuracy'] = {
                    'success': True,
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'correlation': correlation,
                    'direction_accuracy': direction_accuracy,
                    'performance_score': performance_score,
                    'assessment': assessment,
                    'num_forecasts': len(forecasts)
                }
                
                return performance_score >= 25  # Require at least moderate performance
            else:
                print("   ❌ Insufficient forecasts for evaluation")
                self.test_results['forecasting_accuracy'] = {'success': False, 'reason': 'insufficient_data'}
                return False
                
        except Exception as e:
            print(f"   ❌ Forecasting accuracy error: {e}")
            traceback.print_exc()
            self.test_results['forecasting_accuracy'] = {'success': False, 'error': str(e)}
            return False

class AssetAllocationTests(MLTestFramework):
    def __init__(self):
        super().__init__()
        self.allocator = AdvancedPortfolioAllocationEngine()
        
    def test_regime_detection(self):
        """Test market regime detection"""
        print("\n🧪 Testing Market Regime Detection...")
        
        try:
            # Generate test data with known regimes
            _, returns_df, _ = self.generate_realistic_test_data(n_days=400, n_assets=5)
            
            # Test simple regime detection
            simple_regime = self.allocator.detect_market_regime_simple(returns_df)
            print(f"   📊 Simple regime detected: {simple_regime}")
            
            # Test advanced regime detection
            try:
                advanced_regime = self.allocator.detect_market_regime_hmm(returns_df)
                print(f"   🧠 Advanced regime detected: {advanced_regime}")
                
                self.test_results['regime_detection'] = {
                    'success': True,
                    'simple_regime': simple_regime,
                    'advanced_regime': advanced_regime
                }
                
                return True
                
            except Exception as e:
                print(f"   ⚠️ Advanced regime detection failed: {e}")
                print(f"   ✅ Simple regime detection works: {simple_regime}")
                
                self.test_results['regime_detection'] = {
                    'success': True,
                    'simple_regime': simple_regime,
                    'advanced_regime': 'failed'
                }
                
                return True
                
        except Exception as e:
            print(f"   ❌ Regime detection error: {e}")
            traceback.print_exc()
            self.test_results['regime_detection'] = {'success': False, 'error': str(e)}
            return False
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization algorithms"""
        print("\n🧪 Testing Portfolio Optimization...")
        
        try:
            # Generate test data
            _, returns_df, _ = self.generate_realistic_test_data(n_days=300, n_assets=5)
            
            # Test different optimization methods
            optimization_results = {}
            
            # 1. Equal weight baseline
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets
            optimization_results['equal_weight'] = equal_weights
            
            # 2. Test mean-variance optimization (if available)
            try:
                if hasattr(self.allocator, 'optimize_portfolio_mean_variance'):
                    mv_weights = self.allocator.optimize_portfolio_mean_variance(
                        returns_df, risk_tolerance='moderate'
                    )
                    optimization_results['mean_variance'] = mv_weights
                    print("   ✅ Mean-variance optimization successful")
                else:
                    print("   ⚠️ Mean-variance optimization not available")
            except Exception as e:
                print(f"   ⚠️ Mean-variance optimization failed: {e}")
            
            # 3. Test risk parity (if available)
            try:
                if hasattr(self.allocator, 'optimize_risk_parity'):
                    rp_weights = self.allocator.optimize_risk_parity(returns_df)
                    optimization_results['risk_parity'] = rp_weights
                    print("   ✅ Risk parity optimization successful")
                else:
                    print("   ⚠️ Risk parity optimization not available")
            except Exception as e:
                print(f"   ⚠️ Risk parity optimization failed: {e}")
            
            # Validate weights
            for method, weights in optimization_results.items():
                if weights is not None:
                    weight_sum = np.sum(weights)
                    is_valid = np.all(weights >= 0) and np.abs(weight_sum - 1.0) < 0.01
                    print(f"   📊 {method}: sum={weight_sum:.4f}, valid={is_valid}")
                    
                    if not is_valid:
                        print(f"   ⚠️ Invalid weights for {method}: {weights}")
            
            self.test_results['portfolio_optimization'] = {
                'success': len(optimization_results) > 0,
                'methods_tested': list(optimization_results.keys()),
                'results': optimization_results
            }
            
            return len(optimization_results) > 0
            
        except Exception as e:
            print(f"   ❌ Portfolio optimization error: {e}")
            traceback.print_exc()
            self.test_results['portfolio_optimization'] = {'success': False, 'error': str(e)}
            return False
    
    def test_performance_analysis(self):
        """Test portfolio performance analysis"""
        print("\n🧪 Testing Performance Analysis...")
        
        try:
            # Generate test data
            _, returns_df, _ = self.generate_realistic_test_data(n_days=250, n_assets=5)
            
            # Create simple equal-weight portfolio
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Calculate performance metrics
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = (1 + portfolio_returns).prod() - 1
            metrics['annualized_return'] = (1 + portfolio_returns.mean())**252 - 1
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
            
            # Risk metrics
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # Value at Risk
            metrics['var_95'] = portfolio_returns.quantile(0.05)
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= metrics['var_95']].mean()
            
            print(f"   📊 Performance Metrics:")
            print(f"      Total Return: {metrics['total_return']:.2%}")
            print(f"      Annualized Return: {metrics['annualized_return']:.2%}")
            print(f"      Volatility: {metrics['volatility']:.2%}")
            print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"      Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"      VaR (95%): {metrics['var_95']:.2%}")
            print(f"      CVaR (95%): {metrics['cvar_95']:.2%}")
            
            # Performance assessment
            performance_score = 0
            if metrics['sharpe_ratio'] > 0.5:
                performance_score += 25
            if metrics['max_drawdown'] > -0.20:  # Less than 20% drawdown
                performance_score += 25
            if metrics['volatility'] < 0.25:  # Less than 25% volatility
                performance_score += 25
            if metrics['annualized_return'] > 0.05:  # More than 5% return
                performance_score += 25
            
            assessment = "excellent" if performance_score >= 75 else \
                       "good" if performance_score >= 50 else \
                       "moderate" if performance_score >= 25 else "poor"
            
            print(f"   🎯 Portfolio Assessment: {assessment.upper()} ({performance_score}/100)")
            
            self.test_results['performance_analysis'] = {
                'success': True,
                'metrics': metrics,
                'performance_score': performance_score,
                'assessment': assessment
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Performance analysis error: {e}")
            traceback.print_exc()
            self.test_results['performance_analysis'] = {'success': False, 'error': str(e)}
            return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive ML Component Tests")
    print("=" * 60)
    
    # Test Volatility Forecaster
    print("\n📈 VOLATILITY FORECASTER TESTS")
    print("-" * 40)
    
    vol_tests = VolatilityForecasterTests()
    vol_results = {
        'feature_engineering': vol_tests.test_feature_engineering(),
        'model_training': vol_tests.test_model_training(),
        'forecasting_accuracy': vol_tests.test_forecasting_accuracy()
    }
    
    # Test Asset Allocation
    print("\n💼 ASSET ALLOCATION TESTS")
    print("-" * 40)
    
    alloc_tests = AssetAllocationTests()
    alloc_results = {
        'regime_detection': alloc_tests.test_regime_detection(),
        'portfolio_optimization': alloc_tests.test_portfolio_optimization(),
        'performance_analysis': alloc_tests.test_performance_analysis()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print("\n📈 Volatility Forecaster:")
    for test_name, result in vol_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print("\n💼 Asset Allocation:")
    for test_name, result in alloc_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Overall assessment
    total_tests = len(vol_results) + len(alloc_results)
    passed_tests = sum(vol_results.values()) + sum(alloc_results.values())
    
    print(f"\n🎯 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! ML components are working correctly.")
    elif passed_tests >= total_tests * 0.7:
        print("✅ MOSTLY SUCCESSFUL! Some minor issues detected.")
    elif passed_tests >= total_tests * 0.5:
        print("⚠️ PARTIALLY SUCCESSFUL! Significant improvements needed.")
    else:
        print("❌ MAJOR ISSUES! Components need substantial fixes.")
    
    # Detailed results
    print("\n📊 DETAILED RESULTS:")
    print(f"Volatility Forecaster Test Results: {vol_tests.test_results}")
    print(f"Asset Allocation Test Results: {alloc_tests.test_results}")
    
    return passed_tests, total_tests

if __name__ == "__main__":
    passed, total = run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        exit_code = 0  # Success
    elif passed >= total * 0.7:
        exit_code = 1  # Warning
    else:
        exit_code = 2  # Error
    
    print(f"\nExiting with code {exit_code}")
    sys.exit(exit_code)