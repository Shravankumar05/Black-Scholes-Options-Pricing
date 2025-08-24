"""
Comprehensive ML Volatility Forecasting Test Suite
==================================================
This script provides extensive testing and evaluation of the ML volatility forecasting model
including accuracy metrics, backtesting, benchmark comparisons, and actionable insights.
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

from ml_components import AdvancedVolatilityForecaster, EnhancedVolatilityForecaster, VolatilityForecaster
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json

class VolatilityTestFramework:
    def __init__(self):
        self.results = {}
        self.benchmarks = {}
        self.detailed_results = []
        
    def generate_realistic_market_data(self, scenario='normal', days=1000):
        """Generate realistic market data with different volatility regimes"""
        np.random.seed(42)
        
        if scenario == 'normal':
            # Normal market with occasional volatility clusters
            base_vol = 0.16
            vol_persistence = 0.85
            mean_return = 0.08/252
        elif scenario == 'crisis':
            # Crisis period with high volatility
            base_vol = 0.45
            vol_persistence = 0.95
            mean_return = -0.20/252
        elif scenario == 'low_vol':
            # Low volatility bull market
            base_vol = 0.08
            vol_persistence = 0.7
            mean_return = 0.15/252
        elif scenario == 'regime_change':
            # Market with distinct regime changes
            return self._generate_regime_change_data(days)
        
        # GARCH-like volatility process
        volatilities = np.zeros(days)
        volatilities[0] = base_vol
        
        for t in range(1, days):
            shock = np.random.normal(0, 0.05)
            volatilities[t] = (vol_persistence * volatilities[t-1] + 
                             (1 - vol_persistence) * base_vol + shock)
            volatilities[t] = max(0.05, min(1.0, volatilities[t]))  # Bound volatility
        
        # Generate returns with time-varying volatility
        returns = np.zeros(days)
        for t in range(days):
            returns[t] = np.random.normal(mean_return, volatilities[t]/np.sqrt(252))
        
        # Convert to prices
        prices = np.zeros(days)
        prices[0] = 100
        for t in range(1, days):
            prices[t] = prices[t-1] * (1 + returns[t])
        
        dates = pd.date_range('2020-01-01', periods=days, freq='D')
        data = pd.DataFrame({
            'Close': prices,
            'True_Volatility': volatilities,
            'Returns': returns
        }, index=dates)
        
        return data
    
    def _generate_regime_change_data(self, days):
        """Generate data with distinct volatility regimes"""
        np.random.seed(42)
        
        # Define regimes: (volatility, mean_return, duration_range)
        regimes = [
            (0.12, 0.10/252, (100, 200)),  # Low vol, positive returns
            (0.35, -0.15/252, (30, 80)),   # High vol, negative returns
            (0.18, 0.05/252, (80, 150)),   # Medium vol, modest returns
        ]
        
        current_day = 0
        regime_data = []
        
        while current_day < days:
            regime = np.random.choice(len(regimes))
            vol, mean_ret, duration_range = regimes[regime]
            duration = np.random.randint(duration_range[0], duration_range[1])
            
            # Generate data for this regime
            regime_returns = np.random.normal(mean_ret, vol/np.sqrt(252), 
                                            min(duration, days - current_day))
            regime_vol = np.full(len(regime_returns), vol)
            
            regime_data.append({
                'returns': regime_returns,
                'volatility': regime_vol,
                'regime': regime
            })
            
            current_day += len(regime_returns)
        
        # Combine all regimes
        all_returns = np.concatenate([r['returns'] for r in regime_data])
        all_volatilities = np.concatenate([r['volatility'] for r in regime_data])
        
        # Convert to prices
        prices = np.zeros(len(all_returns))
        prices[0] = 100
        for t in range(1, len(all_returns)):
            prices[t] = prices[t-1] * (1 + all_returns[t])
        
        dates = pd.date_range('2020-01-01', periods=len(all_returns), freq='D')
        return pd.DataFrame({
            'Close': prices,
            'True_Volatility': all_volatilities,
            'Returns': all_returns
        }, index=dates)
    
    def load_real_market_data(self, symbols=['AAPL', 'SPY', 'GOOGL', 'MSFT', 'TSLA']):
        """Load real market data for multiple assets"""
        real_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3y")
                if not data.empty:
                    # Calculate realized volatility
                    returns = data['Close'].pct_change().dropna()
                    realized_vol = returns.rolling(20).std() * np.sqrt(252)
                    data['Realized_Volatility'] = realized_vol
                    data['Returns'] = returns
                    real_data[symbol] = data
                    print(f"✓ Loaded data for {symbol}: {len(data)} days")
            except Exception as e:
                print(f"✗ Failed to load {symbol}: {e}")
        
        return real_data
    
    def calculate_baseline_forecasts(self, data, forecast_window=30):
        """Calculate simple baseline forecasts for comparison"""
        returns = data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        baselines = {}
        
        # 1. Historical average
        baselines['historical_avg'] = realized_vol.rolling(60).mean()
        
        # 2. EWMA (Exponentially Weighted Moving Average)
        baselines['ewma'] = realized_vol.ewm(span=30).mean()
        
        # 3. Simple persistence (last observed value)
        baselines['persistence'] = realized_vol.shift(1)
        
        # 4. Trend extrapolation
        trend_forecasts = []
        for i in range(len(realized_vol)):
            if i >= 30:
                recent_vols = realized_vol.iloc[i-30:i]
                if len(recent_vols) > 1:
                    slope, intercept, _, _, _ = stats.linregress(range(len(recent_vols)), recent_vols)
                    forecast = intercept + slope * len(recent_vols)
                    trend_forecasts.append(max(0.05, forecast))
                else:
                    trend_forecasts.append(realized_vol.iloc[i-1] if i > 0 else 0.2)
            else:
                trend_forecasts.append(realized_vol.iloc[i] if i < len(realized_vol) else 0.2)
        
        baselines['trend'] = pd.Series(trend_forecasts, index=realized_vol.index)
        
        return baselines
    
    def comprehensive_backtest(self, data, model_name="ML_Model", is_real_data=False):
        """Perform comprehensive backtesting with walk-forward validation"""
        print(f"\n=== Comprehensive Backtest: {model_name} ===")
        
        # Split data
        train_size = int(len(data) * 0.6)
        val_size = int(len(data) * 0.2)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        print(f"Train: {len(train_data)} days, Val: {len(val_data)} days, Test: {len(test_data)} days")
        
        # Train model using advanced ensemble
        forecaster = AdvancedVolatilityForecaster()
        
        # Adjust parameters based on data size
        if len(train_data) < 200:
            forecaster.lookback_days = min(50, len(train_data) // 3)
            forecaster.forecast_days = min(20, len(train_data) // 6)
        
        train_success = forecaster.train_advanced_ensemble(train_data)
        
        if not train_success:
            print("❌ Advanced model training failed, using enhanced fallback")
            # Create an enhanced fallback forecaster with proper methods
            class EnhancedFallbackForecaster:
                def __init__(self):
                    self.is_trained = True
                
                def train_volatility_model(self, data):
                    return True
                
                def train_advanced_ensemble(self, data):
                    return True
                
                def forecast_volatility(self, data, days_ahead=30):
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 20:
                        return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                    elif len(returns) > 5:
                        return returns.std() * np.sqrt(252)
                    else:
                        return 0.20  # Default volatility
                
                def advanced_ensemble_forecast(self, data, days_ahead=30):
                    return self.forecast_volatility(data, days_ahead)
            
            forecaster = EnhancedFallbackForecaster()
        
        # Get true volatility for comparison
        if is_real_data:
            true_vol = data['Realized_Volatility'].dropna()
        else:
            true_vol = data['True_Volatility']
        
        # Walk-forward testing
        forecasts = []
        dates = []
        window_size = 10  # Reduced window size for more forecasts
        
        # Ensure we don't go beyond available data
        max_index = len(data) - 10  # Reduced safety margin
        test_start = len(val_data)
        test_end = min(len(val_data) + len(test_data) - 10, max_index - train_size)
        
        for i in range(test_start, test_end, window_size):
            # Use expanding window for training
            current_train_end = min(train_size + i - len(val_data), len(data) - 10)
            current_train_data = data.iloc[:current_train_end]
            
            # Retrain model periodically
            if i % (window_size * 5) == 0:  # Retrain every 50 steps
                if hasattr(forecaster, 'train_advanced_ensemble'):
                    forecaster.train_advanced_ensemble(current_train_data)
                elif hasattr(forecaster, 'train_volatility_model'):
                    forecaster.train_volatility_model(current_train_data)
            
            # Generate forecast
            try:
                forecast = forecaster.forecast_volatility(current_train_data, days_ahead=5)
                forecasts.append(forecast)
                
                # Ensure we don't exceed data bounds for date indexing
                forecast_date_idx = min(train_size + i - len(val_data) + 5, len(data) - 1)
                dates.append(data.index[forecast_date_idx])
            except Exception as e:
                print(f"Forecast error at step {i}: {e}")
                # Still try to add a simple forecast
                try:
                    simple_forecast = data['Close'].pct_change().std() * np.sqrt(252)
                    forecasts.append(simple_forecast)
                    safe_date_idx = min(train_size + i - len(val_data), len(data) - 1)
                    dates.append(data.index[safe_date_idx])
                except:
                    continue
        
        # Create forecast series with available data
        if len(forecasts) == 0 or len(dates) == 0:
            print("❌ No valid forecasts generated")
            return None
            
        forecast_series = pd.Series(forecasts, index=dates)
        
        # Align with true volatility - ensure indices exist
        available_indices = true_vol.index.intersection(forecast_series.index)
        if len(available_indices) == 0:
            print("❌ No overlapping data between forecasts and true volatility")
            return None
            
        aligned_true = true_vol[available_indices]
        aligned_forecast = forecast_series[available_indices]
        valid_mask = ~(np.isnan(aligned_true) | np.isnan(aligned_forecast))
        
        aligned_true_clean = aligned_true[valid_mask]
        forecast_clean = aligned_forecast[valid_mask]
        
        if len(aligned_true_clean) < 5:  # Need minimum data points
            print("❌ Insufficient valid predictions for evaluation")
            return None
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(aligned_true_clean, forecast_clean)
        
        # Get baseline comparisons
        baselines = self.calculate_baseline_forecasts(data)
        baseline_metrics = {}
        
        for baseline_name, baseline_series in baselines.items():
            # Safely align baseline with forecast indices
            baseline_available_indices = baseline_series.index.intersection(forecast_series.index)
            
            if len(baseline_available_indices) > 5:  # Need minimum data points
                aligned_baseline = baseline_series[baseline_available_indices]
                aligned_true_baseline = true_vol[baseline_available_indices]
                
                valid_baseline_mask = ~(np.isnan(aligned_true_baseline) | np.isnan(aligned_baseline))
                
                if valid_baseline_mask.sum() > 5:
                    baseline_clean = aligned_baseline[valid_baseline_mask]
                    true_clean_baseline = aligned_true_baseline[valid_baseline_mask]
                    baseline_metrics[baseline_name] = self._calculate_comprehensive_metrics(
                        true_clean_baseline, baseline_clean)
        
        # Store results
        result = {
            'model_name': model_name,
            'data_type': 'Real' if is_real_data else 'Synthetic',
            'n_predictions': len(forecast_clean),
            'metrics': metrics,
            'baseline_metrics': baseline_metrics,
            'forecast_series': forecast_series,
            'true_series': aligned_true,
            'test_period': (dates[0], dates[-1]) if dates else None
        }
        
        self.detailed_results.append(result)
        
        # Print results
        print(f"📊 {len(forecast_clean)} valid predictions generated")
        print(f"📈 R² Score: {metrics['r2']:.4f}")
        print(f"📉 RMSE: {metrics['rmse']:.4f}")
        print(f"🎯 Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        print(f"📊 Hit Rate (±2σ): {metrics['hit_rate']:.2%}")
        
        # Compare with best baseline
        if baseline_metrics:
            best_baseline = max(baseline_metrics.keys(), 
                              key=lambda k: baseline_metrics[k]['r2'])
            best_r2 = baseline_metrics[best_baseline]['r2']
            improvement = metrics['r2'] - best_r2
            print(f"🔄 Best Baseline ({best_baseline}): R² = {best_r2:.4f}")
            print(f"🚀 Improvement: {improvement:+.4f} ({improvement/abs(best_r2)*100:+.1f}%)")
        
        return result
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive forecasting accuracy metrics"""
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_accuracy = 0
        if len(y_true) > 1:
            true_direction = np.diff(y_true)
            pred_direction = np.diff(y_pred)
            direction_matches = np.sum(np.sign(true_direction) == np.sign(pred_direction))
            direction_accuracy = direction_matches / len(true_direction)
        
        # Hit rate (predictions within 2 standard deviations)
        residuals = y_pred - y_true
        threshold = 2 * np.std(residuals)
        hit_rate = np.sum(np.abs(residuals) <= threshold) / len(residuals)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Theil's U statistic
        theil_u = np.sqrt(mse) / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)))
        
        # Forecast bias
        bias = np.mean(y_pred - y_true)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'hit_rate': hit_rate,
            'mape': mape,
            'theil_u': theil_u,
            'bias': bias,
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'std_true': np.std(y_true),
            'std_pred': np.std(y_pred)
        }
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation across multiple scenarios"""
        print("🚀 Starting Comprehensive ML Volatility Evaluation")
        print("=" * 60)
        
        # Test on synthetic data with different scenarios
        scenarios = ['normal', 'crisis', 'low_vol', 'regime_change']
        
        for scenario in scenarios:
            print(f"\n🔬 Testing scenario: {scenario.upper()}")
            data = self.generate_realistic_market_data(scenario, days=1000)  # Reduced from 1500
            result = self.comprehensive_backtest(data, f"ML_Model_{scenario}", False)
            if result:
                self.results[f"synthetic_{scenario}"] = result
        
        # Test on real market data
        print(f"\n📈 Testing on real market data...")
        real_data = self.load_real_market_data()
        
        for symbol, data in real_data.items():
            print(f"\n📊 Testing {symbol}...")
            result = self.comprehensive_backtest(data, f"ML_Model_{symbol}", True)
            if result:
                self.results[f"real_{symbol}"] = result
        
        # Generate comprehensive report
        self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report with actionable insights"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE VOLATILITY FORECASTING EVALUATION REPORT")
        print("="*80)
        
        if not self.results:
            print("❌ No results available for reporting")
            return
        
        # Aggregate metrics
        all_r2 = [r['metrics']['r2'] for r in self.results.values()]
        all_direction_acc = [r['metrics']['direction_accuracy'] for r in self.results.values()]
        all_rmse = [r['metrics']['rmse'] for r in self.results.values()]
        
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY")
        print(f"   Number of test scenarios: {len(self.results)}")
        print(f"   Average R² Score: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})")
        print(f"   Average Direction Accuracy: {np.mean(all_direction_acc):.2%}")
        print(f"   Average RMSE: {np.mean(all_rmse):.4f}")
        
        # Best and worst performers
        best_test = max(self.results.keys(), key=lambda k: self.results[k]['metrics']['r2'])
        worst_test = min(self.results.keys(), key=lambda k: self.results[k]['metrics']['r2'])
        
        print(f"\n🏆 BEST PERFORMANCE: {best_test}")
        best_metrics = self.results[best_test]['metrics']
        print(f"   R² Score: {best_metrics['r2']:.4f}")
        print(f"   Direction Accuracy: {best_metrics['direction_accuracy']:.2%}")
        print(f"   RMSE: {best_metrics['rmse']:.4f}")
        
        print(f"\n📉 WORST PERFORMANCE: {worst_test}")
        worst_metrics = self.results[worst_test]['metrics']
        print(f"   R² Score: {worst_metrics['r2']:.4f}")
        print(f"   Direction Accuracy: {worst_metrics['direction_accuracy']:.2%}")
        print(f"   RMSE: {worst_metrics['rmse']:.4f}")
        
        # Model assessment
        avg_r2 = np.mean(all_r2)
        avg_direction = np.mean(all_direction_acc)
        
        print(f"\n🎯 MODEL ASSESSMENT")
        if avg_r2 >= 0.25 and avg_direction >= 0.6:
            print("   ✅ EXCELLENT: Model shows strong predictive power")
            assessment = "excellent"
        elif avg_r2 >= 0.15 and avg_direction >= 0.55:
            print("   ✅ GOOD: Model has solid forecasting ability")
            assessment = "good"
        elif avg_r2 >= 0.05 and avg_direction >= 0.5:
            print("   ⚠️  MODERATE: Model shows some predictive skill")
            assessment = "moderate"
        else:
            print("   ❌ POOR: Model needs significant improvement")
            assessment = "poor"
        
        # Specific recommendations
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        self._generate_recommendations(assessment, avg_r2, avg_direction)
        
        # Baseline comparison summary
        print(f"\n📈 BASELINE COMPARISON")
        baseline_improvements = []
        for result in self.results.values():
            if 'baseline_metrics' in result and result['baseline_metrics']:
                best_baseline_r2 = max(b['r2'] for b in result['baseline_metrics'].values())
                improvement = result['metrics']['r2'] - best_baseline_r2
                baseline_improvements.append(improvement)
        
        if baseline_improvements:
            avg_improvement = np.mean(baseline_improvements)
            print(f"   Average improvement over baselines: {avg_improvement:+.4f}")
            if avg_improvement > 0.05:
                print("   ✅ Model significantly outperforms simple baselines")
            elif avg_improvement > 0:
                print("   ⚠️  Model marginally outperforms baselines")
            else:
                print("   ❌ Model does not consistently beat simple baselines")
        
        # Save detailed results
        self._save_results_to_file()
        
        print(f"\n💾 Detailed results saved to 'volatility_test_results.json'")
        print("="*80)
    
    def _generate_recommendations(self, assessment, avg_r2, avg_direction):
        """Generate specific recommendations based on performance"""
        
        if assessment == "poor":
            print("   1. 🔧 CRITICAL: Completely redesign the model architecture")
            print("      - Consider LSTM/GRU networks for temporal dependencies")
            print("      - Add more sophisticated features (VIX, options data)")
            print("      - Implement ensemble methods combining multiple models")
            print("   2. 📊 Increase training data quantity and quality")
            print("   3. 🎯 Focus on feature engineering (market regime indicators)")
            print("   4. 🔄 Implement online learning for model adaptation")
            
        elif assessment == "moderate":
            print("   1. 🛠️  Enhance feature engineering:")
            print("      - Add macroeconomic indicators")
            print("      - Include cross-asset correlations")
            print("      - Implement volatility regime detection")
            print("   2. 🎛️  Hyperparameter optimization")
            print("   3. 📈 Consider model ensemble approaches")
            print("   4. 🔍 Analyze prediction errors for systematic patterns")
            
        elif assessment == "good":
            print("   1. 🎯 Fine-tune hyperparameters for marginal improvements")
            print("   2. 🔄 Implement model updating mechanisms")
            print("   3. 📊 Add confidence intervals to predictions")
            print("   4. 🚀 Consider deploying to production with monitoring")
            
        else:  # excellent
            print("   1. ✅ Model is ready for production deployment")
            print("   2. 📊 Implement comprehensive monitoring system")
            print("   3. 🔄 Set up automated retraining pipeline")
            print("   4. 📈 Consider extending to other asset classes")
        
        # Specific technical recommendations
        if avg_direction < 0.55:
            print("   ⚠️  LOW DIRECTION ACCURACY: Add trend-following features")
        
        if avg_r2 < 0.1:
            print("   ⚠️  LOW R²: Consider non-linear models (Random Forest, XGBoost)")
    
    def _save_results_to_file(self):
        """Save detailed results to JSON file"""
        output_data = {}
        
        for test_name, result in self.results.items():
            # Convert numpy arrays and pandas series to lists for JSON serialization
            serializable_result = {
                'model_name': result['model_name'],
                'data_type': result['data_type'],
                'n_predictions': result['n_predictions'],
                'metrics': result['metrics'],
                'test_period': [str(result['test_period'][0]), str(result['test_period'][1])] if result['test_period'] else None
            }
            
            if 'baseline_metrics' in result:
                serializable_result['baseline_metrics'] = result['baseline_metrics']
                
            output_data[test_name] = serializable_result
        
        with open('volatility_test_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

def main():
    """Main execution function"""
    tester = VolatilityTestFramework()
    tester.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()