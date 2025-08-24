import numpy as np
import pandas as pd
import yfinance as yf
from ml_components import vol_forecaster
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    """Generate synthetic price data for testing when real data is not available"""
    np.random.seed(42)
    days = 500
    returns = np.random.normal(0.0005, 0.02, days)  # Mean return 0.05% daily, 2% daily vol
    prices = [100]  # Starting price
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    data.set_index('Date', inplace=True)
    return data

def evaluate_volatility_forecasting(symbol=None, test_size=0.2):
    """
    Evaluate the volatility forecasting model using historical data
    """
    print("=== Volatility Forecasting Model Evaluation ===")
    
    # Get data
    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y")
            print(f"Using real data for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            print("Using synthetic data instead")
            data = generate_synthetic_data()
    else:
        print("Using synthetic data")
        data = generate_synthetic_data()
    
    if data.empty:
        print("No data available")
        return
    
    print(f"Data shape: {data.shape}")
    
    # Split data for backtesting
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    
    # Train the model
    print("Training volatility model...")
    train_success = vol_forecaster.train_volatility_model(train_data)
    
    if not train_success:
        print("Failed to train model")
        return
    
    print("Model trained successfully")
    
    # Calculate actual volatility for test period
    test_returns = test_data['Close'].pct_change().dropna()
    actual_vol = test_returns.rolling(20).std() * np.sqrt(252)  # Annualized 20-day vol
    actual_vol = actual_vol.dropna()
    
    if len(actual_vol) == 0:
        print("Not enough data to calculate actual volatility")
        return
    
    # Generate forecasts
    print("Generating forecasts...")
    forecasts = []
    forecast_dates = []
    
    # Walk forward testing
    for i in range(20, len(test_data)):
        current_data = pd.concat([train_data, test_data.iloc[:i]])
        try:
            forecast = vol_forecaster.forecast_volatility(current_data, days_ahead=5)
            forecasts.append(forecast)
            forecast_dates.append(test_data.index[i])
        except Exception as e:
            print(f"Error in forecast at index {i}: {e}")
            forecasts.append(np.nan)
            forecast_dates.append(test_data.index[i])
    
    # Align forecasts with actual values
    forecast_series = pd.Series(forecasts, index=forecast_dates)
    aligned_actual = actual_vol[forecast_series.index]
    aligned_forecast = forecast_series.dropna()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(aligned_actual) | np.isnan(aligned_forecast))
    actual_valid = aligned_actual[valid_mask]
    forecast_valid = aligned_forecast[valid_mask]
    
    if len(actual_valid) == 0:
        print("No valid data points for comparison")
        return
    
    # Ensure we have the same number of points for direction accuracy
    min_len = min(len(actual_valid), len(forecast_valid))
    if min_len < 2:
        print("Not enough data points for direction accuracy calculation")
        direction_accuracy = 0
    else:
        actual_for_direction = actual_valid.iloc[:min_len]
        forecast_for_direction = forecast_valid.iloc[:min_len]
        
        # Calculate direction accuracy (whether both forecast and actual move in same direction)
        actual_direction = np.diff(actual_for_direction)
        forecast_direction = np.diff(forecast_for_direction)
        
        # Make sure arrays are the same length
        min_direction_len = min(len(actual_direction), len(forecast_direction))
        actual_direction = actual_direction[:min_direction_len]
        forecast_direction = forecast_direction[:min_direction_len]
        
        direction_matches = np.sum(np.sign(actual_direction) == np.sign(forecast_direction))
        direction_accuracy = direction_matches / len(actual_direction) if len(actual_direction) > 0 else 0
    
    # Calculate metrics using only valid data
    mse = mean_squared_error(actual_valid, forecast_valid)
    mae = mean_absolute_error(actual_valid, forecast_valid)
    
    # For R2, we need to make sure arrays are the same length
    min_r2_len = min(len(actual_valid), len(forecast_valid))
    actual_r2 = actual_valid.iloc[:min_r2_len]
    forecast_r2 = forecast_valid.iloc[:min_r2_len]
    r2 = r2_score(actual_r2, forecast_r2)
    
    print("\n=== Evaluation Results ===")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print(f"Number of predictions: {len(forecast_valid)}")
    
    # Summary
    if r2 > 0.3:
        print("\n✅ Model performance is GOOD (R² > 0.3)")
    elif r2 > 0.1:
        print("\n⚠️ Model performance is MODERATE (0.1 < R² < 0.3)")
    else:
        print("\n❌ Model performance is POOR (R² < 0.1)")
    
    # Show sample predictions
    print("\n=== Sample Predictions ===")
    sample_size = min(10, len(actual_valid))
    comparison_df = pd.DataFrame({
        'Actual Volatility': actual_valid.tail(sample_size),
        'Forecasted Volatility': forecast_valid.tail(sample_size)
    })
    print(comparison_df)

def test_model_sensitivity():
    """Test how the model performs with different market conditions"""
    print("\n=== Model Sensitivity Test ===")
    
    # Test with high volatility data
    print("Testing with high volatility synthetic data...")
    np.random.seed(42)
    days = 300
    # Higher volatility returns
    returns = np.random.normal(0.0005, 0.04, days)  # 4% daily vol
    prices = [100]
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    high_vol_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=days, freq='D'),
        'Close': prices
    })
    high_vol_data.set_index('Date', inplace=True)
    
    success = vol_forecaster.train_volatility_model(high_vol_data)
    if success:
        forecast = vol_forecaster.forecast_volatility(high_vol_data)
        actual_vol = high_vol_data['Close'].pct_change().dropna().std() * np.sqrt(252)
        print(f"High volatility period - Actual: {actual_vol:.4f}, Forecast: {forecast:.4f}")
    else:
        print("Failed to train on high volatility data")

if __name__ == "__main__":
    print("Testing ML Volatility Forecasting Model")
    print("=" * 50)
    
    # Test with synthetic data
    evaluate_volatility_forecasting()
    
    # Test with real stock data if available
    evaluate_volatility_forecasting("AAPL")
    
    # Test model sensitivity
    test_model_sensitivity()
    
    print("\n=== Test Complete ===")