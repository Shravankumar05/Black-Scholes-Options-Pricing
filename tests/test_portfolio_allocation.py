import unittest
import numpy as np
import pandas as pd
from portfolio_allocation import PortfolioAllocationEngine

class TestPortfolioAllocation(unittest.TestCase):

    def setUp(self):
        """Set up a mock PortfolioAllocation instance with synthetic data."""
        self.assets = ['AAPL', 'GOOG', 'MSFT']
        self.price_data = self._generate_synthetic_data(self.assets)
        self.returns_data = self.price_data.pct_change().dropna()
        self.pa = PortfolioAllocationEngine()

    def _generate_synthetic_data(self, assets):
        """Generates synthetic historical data for testing."""
        dates = pd.date_range('2022-01-01', periods=252)
        data = pd.DataFrame(index=dates)
        for asset in assets:
            # Introduce some correlation and trend
            base_returns = np.random.randn(252) * 0.015 + (0.0005 * (assets.index(asset) + 1))
            price = 100 * (1 + base_returns).cumprod()
            data[asset] = price
        return data

    def test_allocation_execution(self):
        """Test that the main allocation method runs without errors and returns a valid allocation."""
        # Using regime_adaptive_allocation as the main entry point for testing
        allocation = self.pa.regime_adaptive_allocation(self.returns_data, self.assets)

        # Check if allocation is a dictionary
        self.assertIsInstance(allocation, dict)
        
        # Check if weights sum to approximately 1
        self.assertAlmostEqual(sum(allocation.values()), 1.0, places=5)

        # Check for non-negative weights
        for weight in allocation.values():
            self.assertGreaterEqual(weight, 0)

    def test_ml_model_integration(self):
        """Test that the ML model for return prediction is being used."""
        # This test checks if the ML-enhanced predictions are different from simple historical returns
        self.pa.train_return_predictor(self.returns_data)
        ml_predicted_returns = self.pa.predict_expected_returns(self.returns_data)

        self.assertIsNotNone(ml_predicted_returns)
        self.assertEqual(len(ml_predicted_returns), len(self.assets))

        # A simple check to ensure the model is doing something beyond historical average
        historical_mean_returns = self.returns_data.mean() * 252
        self.assertFalse(np.allclose(ml_predicted_returns.values, historical_mean_returns.values, atol=1e-5),
                         "ML predictions should not be identical to historical mean returns.")

if __name__ == '__main__':
    unittest.main()

