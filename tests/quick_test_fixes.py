"""
Quick test to validate the fixes for the volatility model evaluation
"""

import sys
import os
import traceback

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_volatility_model_fix():
    """Test if the volatility model index error is fixed"""
    print("🧪 Testing volatility model fixes...")
    
    try:
        from test_ml_volatility_comprehensive import VolatilityTestFramework
        
        # Create test framework
        tester = VolatilityTestFramework()
        
        # Generate small dataset for testing
        print("📊 Generating test data...")
        test_data = tester.generate_realistic_market_data('normal', days=300)
        
        print("🔬 Running quick backtest...")
        result = tester.comprehensive_backtest(test_data, "Quick_Test", False)
        
        if result:
            print("✅ Volatility model test PASSED!")
            print(f"   Generated {result['n_predictions']} predictions")
            print(f"   R² Score: {result['metrics']['r2']:.4f}")
            return True
        else:
            print("❌ Volatility model test returned None")
            return False
            
    except Exception as e:
        print(f"❌ Volatility model test FAILED: {e}")
        traceback.print_exc()
        return False

def test_portfolio_model():
    """Test if portfolio models work correctly"""
    print("\n🧪 Testing portfolio model...")
    
    try:
        from test_portfolio_comprehensive import PortfolioTestFramework
        
        # Create test framework
        tester = PortfolioTestFramework()
        
        # Generate small dataset for testing
        print("📊 Generating portfolio test data...")
        assets = ['AAPL', 'MSFT', 'GOOGL']
        _, returns_data = tester.generate_realistic_asset_data(assets, 'normal', days=200)
        
        print("🔬 Running quick portfolio backtest...")
        strategy_results, analysis = tester.comprehensive_strategy_backtest(
            returns_data, assets, "Quick_Portfolio_Test"
        )
        
        if strategy_results and analysis:
            print("✅ Portfolio model test PASSED!")
            print(f"   Best strategy: {analysis['best_strategy']}")
            print(f"   Best Sharpe: {analysis['best_strategy_metrics']['sharpe_ratio']:.4f}")
            return True
        else:
            print("❌ Portfolio model test returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Portfolio model test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run quick tests"""
    print("🚀 Running Quick Fix Validation Tests")
    print("=" * 50)
    
    vol_passed = test_volatility_model_fix()
    portfolio_passed = test_portfolio_model()
    
    print("\n" + "=" * 50)
    print("📋 QUICK TEST RESULTS")
    print("=" * 50)
    
    print(f"Volatility Model: {'✅ PASSED' if vol_passed else '❌ FAILED'}")
    print(f"Portfolio Model: {'✅ PASSED' if portfolio_passed else '❌ FAILED'}")
    
    if vol_passed and portfolio_passed:
        print("\n🎉 All fixes working correctly!")
        print("You can now run: python tests/run_comprehensive_evaluation.py")
    elif vol_passed:
        print("\n⚠️ Volatility model fixed, but portfolio needs work")
    elif portfolio_passed:
        print("\n⚠️ Portfolio model working, but volatility needs more fixes")
    else:
        print("\n❌ Both models need additional fixes")
    
    return vol_passed and portfolio_passed

if __name__ == "__main__":
    success = main()
    print(f"\nQuick test {'PASSED' if success else 'FAILED'}")