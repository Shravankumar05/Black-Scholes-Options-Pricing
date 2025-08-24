import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class OptionsMLPredictor:
    def __init__(self):
        self.iv_model = None
        self.price_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, price_data, options_data=None):
        features = []
        
        if len(price_data) >= 20:
            price_data['sma_20'] = price_data['Close'].rolling(20).mean()
            price_data['sma_50'] = price_data['Close'].rolling(50).mean() if len(price_data) >= 50 else price_data['sma_20']
            price_data['rsi'] = self._calculate_rsi(price_data['Close'])
            price_data['volatility'] = price_data['Close'].pct_change().rolling(20).std()
            
            price_data['price_change_1d'] = price_data['Close'].pct_change()
            price_data['price_change_5d'] = price_data['Close'].pct_change(5)
            price_data['price_change_20d'] = price_data['Close'].pct_change(20)
            
            price_data['vol_ratio'] = price_data['volatility'] / price_data['volatility'].rolling(60).mean()
            
            feature_columns = ['sma_20', 'sma_50', 'rsi', 'volatility', 
                             'price_change_1d', 'price_change_5d', 'price_change_20d', 'vol_ratio']
            
            return price_data[feature_columns].dropna()
        
        return pd.DataFrame()
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_implied_volatility_model(self, iv_surface_df):
        if iv_surface_df.empty:
            return False
        
        try:
            features = ['moneyness', 'time_to_expiry', 'volume', 'open_interest']
            available_features = [f for f in features if f in iv_surface_df.columns]
            
            if len(available_features) < 2:
                return False
            
            X = iv_surface_df[available_features].fillna(0)
            y = iv_surface_df['implied_vol']
            
            q1, q3 = y.quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (y >= q1 - 1.5*iqr) & (y <= q3 + 1.5*iqr)
            X, y = X[mask], y[mask]
            
            if len(X) < 10:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear': LinearRegression()
            }
            
            best_score = -np.inf
            best_model = None
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
            
            if best_model is not None:
                self.iv_model = best_model
                self.scaler.fit(X)
                self.is_trained = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Error training IV model: {e}")
            return False
    
    def predict_implied_volatility(self, moneyness, time_to_expiry, volume=1000, open_interest=500):
        if not self.is_trained or self.iv_model is None:
            return 0.25
        
        try:
            features = np.array([[moneyness, time_to_expiry, volume, open_interest]])
            prediction = self.iv_model.predict(features)[0]
            return max(0.05, min(2.0, prediction))
        except:
            return 0.25

class VolatilityForecaster:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.lookback_days = 60
        self.forecast_days = 30
    
    def prepare_volatility_features(self, price_data):
        if len(price_data) < self.lookback_days:
            return None, None
        
        returns = price_data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        features = []
        targets = []
        
        for i in range(self.lookback_days, len(realized_vol) - self.forecast_days):
            feature_window = realized_vol.iloc[i-self.lookback_days:i].values
            return_window = returns.iloc[i-20:i].values
            
            combined_features = np.concatenate([feature_window, return_window])
            features.append(combined_features)
            
            future_vol = realized_vol.iloc[i:i+self.forecast_days].mean()
            targets.append(future_vol)
        
        return np.array(features), np.array(targets)
    
    def train_volatility_model(self, price_data):
        X, y = self.prepare_volatility_features(price_data)
        
        if X is None or len(X) < 20:
            return False
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            score = self.model.score(X_test_scaled, y_test)
            
            return score > 0.1
            
        except Exception as e:
            print(f"Error training volatility model: {e}")
            return False
    
    def forecast_volatility(self, price_data, days_ahead=30):
        if self.model is None:
            returns = price_data['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)
        
        try:
            returns = price_data['Close'].pct_change().dropna()
            realized_vol = returns.rolling(20).std() * np.sqrt(252)
            
            latest_vol = realized_vol.tail(self.lookback_days).values
            latest_returns = returns.tail(20).values
            features = np.concatenate([latest_vol, latest_returns]).reshape(1, -1)
            
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            return max(0.05, min(2.0, prediction))
            
        except Exception as e:
            print(f"Error forecasting volatility: {e}")
            returns = price_data['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)

class MarketRegimeDetector:
    def __init__(self):
        self.regimes = ['Bull Market', 'Bear Market', 'Sideways Market', 'High Volatility']
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def detect_current_regime(self, price_data, returns_data=None):
        try:
            if len(price_data) < 60:
                return "Insufficient Data"
            
            returns = price_data['Close'].pct_change().dropna()
            
            sma_20 = price_data['Close'].rolling(20).mean()
            sma_50 = price_data['Close'].rolling(50).mean()
            current_price = price_data['Close'].iloc[-1]
            
            volatility = returns.rolling(20).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1]
            
            price_change_20d = (current_price - price_data['Close'].iloc[-21]) / price_data['Close'].iloc[-21]
            
            if current_vol > 0.35:
                return "High Volatility"
            elif price_change_20d > 0.05 and current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                return "Bull Market"
            elif price_change_20d < -0.05 and current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                return "Bear Market"
            else:
                return "Sideways Market"
                
        except Exception as e:
            print(f"Error detecting regime: {e}")
            return "Unknown"

ml_predictor = OptionsMLPredictor()
vol_forecaster = VolatilityForecaster()
regime_detector = MarketRegimeDetector()