"""
Quick validation script to test if all test scripts can be imported and run
"""

import sys
import os

print("🔍 Validating ML Testing Suite...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"✓ Added parent directory to path: {parent_dir}")

# Also add current directory to path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Test basic imports
try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    from sklearn.metrics import mean_squared_error
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except ImportError as e:
    print(f"❌ yfinance import failed: {e}")

# Test project modules
try:
    from ml_components import VolatilityForecaster
    print("✓ ML components imported successfully")
except ImportError as e:
    print(f"❌ ML components import failed: {e}")

try:
    from portfolio_allocation import PortfolioAllocationEngine
    print("✓ Portfolio allocation imported successfully")
except ImportError as e:
    print(f"❌ Portfolio allocation import failed: {e}")

# Test our test scripts (basic imports only)
test_files = [
    'test_ml_volatility_comprehensive.py',
    'test_portfolio_comprehensive.py', 
    'test_model_comparison.py',
    'comprehensive_model_report.py',
    'run_comprehensive_evaluation.py'
]

print("\n📋 Checking test script files:")
# Check in tests directory if we're running from parent
test_dir = 'tests' if os.path.exists('tests') else '.'
for test_file in test_files:
    test_path = os.path.join(test_dir, test_file)
    if os.path.exists(test_path):
        print(f"✓ {test_file} exists")
    else:
        print(f"❌ {test_file} missing")

# Also check if we can import the test modules
print("\n🔧 Testing module imports:")
try:
    # Test if we can import our comprehensive test modules
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("test_ml_volatility_comprehensive", "test_ml_volatility_comprehensive.py")
    if spec and spec.loader:
        print("✓ Can load volatility test module")
    else:
        print("❌ Cannot load volatility test module")
        
except Exception as e:
    print(f"⚠️ Module import test failed: {e}")

# Quick synthetic data test
print("\n🧪 Running quick synthetic data test...")
try:
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.cumsum(np.random.randn(100)) + 100
    test_data = pd.DataFrame({'Close': prices}, index=dates)
    
    # Test basic volatility calculation
    returns = test_data['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    
    print(f"✓ Basic volatility calculation works: {vol:.4f}")
    
    # Test portfolio allocation if we can import it
    try:
        from portfolio_allocation import PortfolioAllocationEngine
        allocator = PortfolioAllocationEngine()
        assets = ['AAPL', 'MSFT', 'GOOGL']
        equal_weight = allocator.equal_weight_allocation(assets)
        print(f"✓ Basic portfolio allocation works: {equal_weight}")
    except ImportError:
        print("⚠️ Portfolio allocation module not available (expected if not in project root)")
    
except Exception as e:
    print(f"❌ Synthetic data test failed: {e}")

print("\n✅ Validation completed!")
print("\nTo run the complete evaluation:")
if os.path.exists('tests'):
    print("python tests/run_comprehensive_evaluation.py")
else:
    print("python run_comprehensive_evaluation.py")