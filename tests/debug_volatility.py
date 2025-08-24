"""
Debug script to trace volatility model issues
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_components import VolatilityForecaster

def debug_volatility_forecaster():
    """Debug the volatility forecaster step by step"""
    print("🔍 Debugging Volatility Forecaster...")
    
    # Generate simple test data
    np.random.seed(42)
    days = 300
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.02, days)
    prices = [100]
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    data = pd.DataFrame({'Close': prices}, index=dates)
    
    print(f"📊 Generated data: {len(data)} days")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Test volatility forecaster
    forecaster = VolatilityForecaster()
    print(f"📐 Default parameters: lookback={forecaster.lookback_days}, forecast={forecaster.forecast_days}")
    
    # Test data preparation
    X, y = forecaster.prepare_volatility_features(data)
    
    if X is None:
        print("❌ Feature preparation failed - insufficient data")
        print(f"   Need at least {forecaster.lookback_days} days for lookback")
        return False
    
    print(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test model training
    print("🎯 Training model...")
    success = forecaster.train_volatility_model(data)
    
    if not success:
        print("❌ Model training failed")
        
        # Try with smaller parameters
        print("🔧 Trying with smaller parameters...")
        forecaster.lookback_days = 30
        forecaster.forecast_days = 15
        
        X, y = forecaster.prepare_volatility_features(data)
        if X is not None:
            print(f"✅ Smaller features: {X.shape[0]} samples, {X.shape[1]} features")
            success = forecaster.train_volatility_model(data)
        
        if not success:
            print("❌ Even smaller parameters failed")
            return False
    
    print("✅ Model training succeeded!")
    
    # Test forecasting
    print("📈 Testing forecasting...")
    try:
        forecast = forecaster.forecast_volatility(data)
        print(f"✅ Forecast generated: {forecast:.4f}")
        return True
    except Exception as e:
        print(f"❌ Forecasting failed: {e}")
        return False

def debug_comprehensive_backtest():
    """Debug the comprehensive backtest process"""
    print("\n🔍 Debugging Comprehensive Backtest...")
    
    from test_ml_volatility_comprehensive import VolatilityTestFramework
    
    tester = VolatilityTestFramework()
    
    # Generate data with verbose output
    print("📊 Generating market data...")
    data = tester.generate_realistic_market_data('normal', days=400)
    print(f"   Generated {len(data)} days of data")
    
    if 'True_Volatility' in data.columns:
        print(f"   True volatility range: {data['True_Volatility'].min():.4f} - {data['True_Volatility'].max():.4f}")
    
    # Test train/val/test split
    train_size = int(len(data) * 0.6)
    val_size = int(len(data) * 0.2)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    
    print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Test forecaster with this data
    forecaster = VolatilityForecaster()
    
    # Adjust parameters
    if len(train_data) < 200:
        forecaster.lookback_days = min(30, len(train_data) // 4)
        forecaster.forecast_days = min(15, len(train_data) // 8)
        print(f"🔧 Adjusted parameters: lookback={forecaster.lookback_days}, forecast={forecaster.forecast_days}")
    
    success = forecaster.train_volatility_model(train_data)
    print(f"🎯 Model training: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        # Test walk-forward
        print("📈 Testing walk-forward forecasting...")
        
        forecasts = []
        for i in range(len(val_data), min(len(val_data) + 30, len(data) - 30), 10):
            try:
                current_train_data = data.iloc[:train_size + i - len(val_data)]
                forecast = forecaster.forecast_volatility(current_train_data)
                forecasts.append(forecast)
                print(f"   Step {i}: forecast = {forecast:.4f}")
            except Exception as e:
                print(f"   Step {i}: ERROR = {e}")
                break
        
        print(f"✅ Generated {len(forecasts)} forecasts")
        return len(forecasts) > 0
    
    return False

def main():
    print("🚀 Volatility Model Debug Session")
    print("=" * 50)
    
    step1 = debug_volatility_forecaster()
    step2 = debug_comprehensive_backtest()
    
    print("\n" + "=" * 50)
    print("📋 DEBUG RESULTS")
    print("=" * 50)
    
    print(f"Basic Forecaster: {'✅ OK' if step1 else '❌ FAILED'}")
    print(f"Comprehensive Test: {'✅ OK' if step2 else '❌ FAILED'}")
    
    if step1 and step2:
        print("\n🎉 Volatility model debugging complete - should work now!")
    else:
        print("\n🔧 Issues found - need more fixes")

if __name__ == "__main__":
    main()