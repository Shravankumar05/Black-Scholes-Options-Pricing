import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

class AdvancedVolatilityForecaster:
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_scaler = StandardScaler()
        self.lookback_days = 100  # Longer lookback for pattern recognition
        self.forecast_days = 30
        self.lstm_model = None
        self.garch_params = {}
        self.is_trained = False
        self.feature_names = []
        
        # Advanced parameters
        self.lstm_units = 64
        self.attention_heads = 4
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        
    def create_advanced_features(self, price_data):
        """Create 50+ advanced features for volatility prediction"""
        try:
            if len(price_data) < self.lookback_days:
                return None
                
            features_df = pd.DataFrame(index=price_data.index)
            returns = price_data['Close'].pct_change().dropna()
            prices = price_data['Close']
            
            # === BASIC VOLATILITY FEATURES ===
            for window in [5, 10, 20, 30, 60]:
                features_df[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
                try:
                    features_df[f'vol_skew_{window}d'] = returns.rolling(window).skew()
                    # Use apply for kurtosis as rolling().kurtosis() may not be available
                    features_df[f'vol_kurt_{window}d'] = returns.rolling(window).apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0)
                except:
                    features_df[f'vol_skew_{window}d'] = 0
                    features_df[f'vol_kurt_{window}d'] = 0
            
            # === VOLATILITY CLUSTERING ===
            abs_returns = np.abs(returns)
            for window in [10, 20, 50]:
                features_df[f'abs_ret_ma_{window}d'] = abs_returns.rolling(window).mean()
                features_df[f'vol_clustering_{window}d'] = abs_returns.rolling(window).std()
            
            # === PRICE-BASED FEATURES ===
            for window in [5, 10, 20, 50]:
                features_df[f'sma_{window}'] = prices.rolling(window).mean()
                features_df[f'price_position_{window}'] = prices / features_df[f'sma_{window}'] - 1
                features_df[f'bb_position_{window}'] = self._bollinger_position(prices, window)
                
            # === MOMENTUM FEATURES ===
            for window in [5, 10, 20, 60]:
                features_df[f'momentum_{window}d'] = prices.pct_change(window)
                features_df[f'rsi_{window}d'] = self._calculate_advanced_rsi(prices, window)
                
            # === TECHNICAL INDICATORS ===
            features_df['macd'] = self._calculate_macd(prices)
            features_df['macd_signal'] = self._calculate_macd_signal(prices)
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # === VOLATILITY REGIME FEATURES ===
            features_df['vol_regime'] = self._detect_volatility_regime(returns)
            features_df['vol_trend'] = self._calculate_volatility_trend(returns)
            features_df['vol_mean_reversion'] = self._calculate_mean_reversion_strength(returns)
            
            # === HIGHER-ORDER MOMENTS ===
            for window in [20, 60]:
                roll_rets = returns.rolling(window)
                try:
                    features_df[f'skewness_{window}d'] = roll_rets.skew()
                    features_df[f'kurtosis_{window}d'] = roll_rets.apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0)
                except:
                    features_df[f'skewness_{window}d'] = 0
                    features_df[f'kurtosis_{window}d'] = 0
                features_df[f'downside_vol_{window}d'] = roll_rets.apply(lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 0 else 0)
                
            # === JUMP DETECTION ===
            features_df['jump_indicator'] = self._detect_price_jumps(returns)
            features_df['gap_indicator'] = self._detect_gaps(price_data)
            
            # === VOLUME PROXIES (using price-based proxies) ===
            if 'Volume' in price_data.columns:
                volume = price_data['Volume']
                features_df['volume_trend'] = volume.pct_change(20)
                features_df['price_volume_trend'] = returns.rolling(20).corr(volume.pct_change())
            else:
                # Use absolute returns as volume proxy
                features_df['volume_proxy'] = abs_returns.rolling(20).mean()
                features_df['volume_trend'] = features_df['volume_proxy'].pct_change(10)
            
            # === CROSS-SECTIONAL FEATURES (if multiple assets) ===
            features_df['intraday_range'] = self._calculate_intraday_range(price_data)
            features_df['overnight_return'] = self._calculate_overnight_returns(price_data)
            
            # === TIME-BASED FEATURES ===
            features_df['day_of_week'] = price_data.index.dayofweek
            features_df['month_of_year'] = price_data.index.month
            features_df['is_month_end'] = (price_data.index.day > 25).astype(int)
            
            # Store feature names
            self.feature_names = [col for col in features_df.columns if col != 'target']
            
            return features_df.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            print(f"Feature creation error: {e}")
            return None
        
    def _bollinger_position(self, prices, window):
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (prices - lower) / (upper - lower)
    
    def _calculate_advanced_rsi(self, prices, window):
        """Calculate RSI with proper handling"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_macd_signal(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD signal line"""
        macd = self._calculate_macd(prices, fast, slow)
        return macd.ewm(span=signal).mean()
    
    def _detect_volatility_regime(self, returns):
        """Advanced volatility regime detection"""
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        
        regime = pd.Series(0, index=returns.index)
        regime[vol_20d > vol_60d * 1.5] = 2  # High vol regime
        regime[vol_20d < vol_60d * 0.7] = 1  # Low vol regime
        return regime
    
    def _calculate_volatility_trend(self, returns):
        """Calculate volatility trend"""
        vol_5d = returns.rolling(5).std()
        vol_20d = returns.rolling(20).std()
        return (vol_5d / vol_20d - 1).fillna(0)
    
    def _calculate_mean_reversion_strength(self, returns):
        """Calculate mean reversion strength using Hurst exponent approximation"""
        def hurst_approx(series, max_lag=20):
            if len(series) < max_lag * 2:
                return 0.5
            try:
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        hurst_values = returns.rolling(60).apply(lambda x: hurst_approx(x.values))
        return hurst_values.fillna(0.5)
    
    def _detect_price_jumps(self, returns, threshold=3):
        """Detect price jumps using z-score"""
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        z_scores = (returns - rolling_mean) / rolling_std
        return (np.abs(z_scores) > threshold).astype(int)
    
    def _detect_gaps(self, price_data):
        """Detect overnight gaps"""
        if 'Open' in price_data.columns:
            prev_close = price_data['Close'].shift(1)
            gap_return = (price_data['Open'] - prev_close) / prev_close
            return (np.abs(gap_return) > 0.02).astype(int)  # 2% gap threshold
        else:
            return pd.Series(0, index=price_data.index)
    
    def _calculate_intraday_range(self, price_data):
        """Calculate intraday price range"""
        if 'High' in price_data.columns and 'Low' in price_data.columns:
            return (price_data['High'] - price_data['Low']) / price_data['Close']
        else:
            # Proxy using rolling max/min of close prices
            high_proxy = price_data['Close'].rolling(5).max()
            low_proxy = price_data['Close'].rolling(5).min()
            return (high_proxy - low_proxy) / price_data['Close']
    
    def _calculate_overnight_returns(self, price_data):
        """Calculate overnight returns"""
        if 'Open' in price_data.columns:
            prev_close = price_data['Close'].shift(1)
            return (price_data['Open'] - prev_close) / prev_close
        else:
            return pd.Series(0, index=price_data.index)
    def create_lstm_model(self, input_shape):
        """Create advanced LSTM model with attention mechanism"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        try:
            # Input layer
            inputs = tf.keras.Input(shape=input_shape)
            
            # First LSTM layer with return sequences
            lstm1 = LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(inputs)
            lstm1 = LayerNormalization()(lstm1)
            
            # Second LSTM layer
            lstm2 = LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_rate)(lstm1)
            lstm2 = LayerNormalization()(lstm2)
            
            # Attention mechanism (simplified)
            attention_weights = Dense(1, activation='softmax')(lstm2)
            attended = tf.keras.layers.Multiply()([lstm2, attention_weights])
            attended = tf.keras.layers.GlobalAveragePooling1D()(attended)
            
            # Dense layers
            dense1 = Dense(32, activation='relu')(attended)
            dense1 = Dropout(self.dropout_rate)(dense1)
            
            dense2 = Dense(16, activation='relu')(dense1)
            dense2 = Dropout(self.dropout_rate / 2)(dense2)
            
            # Output layer
            outputs = Dense(1, activation='linear')(dense2)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with advanced optimizer
            optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
            
        except Exception as e:
            print(f"LSTM model creation failed: {e}")
            return None
    
    def prepare_lstm_data(self, features_df, target_col, sequence_length=20):
        """Prepare data for LSTM training"""
        try:
            # Remove non-numeric columns
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                return None, None, None, None
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(numeric_features)):
                X_sequences.append(numeric_features.iloc[i-sequence_length:i].values)
                y_sequences.append(numeric_features.iloc[i][target_col])
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"LSTM data preparation failed: {e}")
            return None, None, None, None
    
    def fit_garch_component(self, returns):
        """Fit GARCH component for volatility clustering"""
        try:
            # Simple GARCH(1,1) parameter estimation
            squared_returns = returns ** 2
            
            # Initialize parameters
            omega = squared_returns.var() * 0.1
            alpha = 0.1
            beta = 0.8
            
            # Simple parameter update (could be replaced with MLE)
            long_run_var = squared_returns.mean()
            
            self.garch_params = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'long_run_var': long_run_var
            }
            
            return True
            
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            return False
    def train_advanced_ensemble(self, price_data):
        """Train sophisticated ensemble with LSTM, XGBoost, and traditional models"""
        try:
            if 'Close' not in price_data.columns or len(price_data) < self.lookback_days + 50:
                return False
            
            print("🚀 Training Advanced ML Ensemble...")
            
            # Create advanced features
            features_df = self.create_advanced_features(price_data)
            if features_df is None:
                return False
            
            # Add target variable (future volatility)
            returns = price_data['Close'].pct_change().dropna()
            realized_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Create forward-looking targets
            targets = []
            for i in range(len(realized_vol) - self.forecast_days):
                future_vol = realized_vol.iloc[i:i+self.forecast_days].mean()
                targets.append(future_vol)
            
            # Align features and targets
            feature_data = features_df.iloc[:len(targets)]
            target_data = np.array(targets)
            
            if len(feature_data) < 100:  # Need sufficient data
                return False
            
            # Split data temporally (no shuffling for time series)
            split_idx = int(len(feature_data) * 0.7)
            
            X_train = feature_data.iloc[:split_idx]
            X_test = feature_data.iloc[split_idx:]
            y_train = target_data[:split_idx]
            y_test = target_data[split_idx:]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train.fillna(0))
            X_test_scaled = self.feature_scaler.transform(X_test.fillna(0))
            
            model_scores = {}
            
            # === 1. LSTM MODEL ===
            if TENSORFLOW_AVAILABLE:
                print("   📊 Training LSTM model...")
                try:
                    # Prepare LSTM data
                    sequence_length = 20
                    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = self.prepare_lstm_data(
                        feature_data.iloc[:split_idx], 'target', sequence_length
                    )
                    
                    if lstm_X_train is not None:
                        # Create and train LSTM
                        lstm_model = self.create_lstm_model((sequence_length, lstm_X_train.shape[2]))
                        
                        if lstm_model is not None:
                            # Add target to feature data for LSTM
                            feature_data_with_target = feature_data.copy()
                            feature_data_with_target['target'] = np.concatenate([target_data, [target_data[-1]] * (len(feature_data) - len(target_data))])
                            
                            # Re-prepare with target
                            lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = self.prepare_lstm_data(
                                feature_data_with_target.iloc[:split_idx], 'target', sequence_length
                            )
                            
                            if lstm_X_train is not None and len(lstm_y_train) > 0:
                                # Early stopping and learning rate reduction
                                callbacks = [
                                    EarlyStopping(patience=10, restore_best_weights=True),
                                    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                                ]
                                
                                # Train LSTM
                                history = lstm_model.fit(
                                    lstm_X_train, lstm_y_train,
                                    epochs=50, batch_size=32,
                                    validation_split=0.2,
                                    callbacks=callbacks,
                                    verbose=0
                                )
                                
                                # Evaluate
                                if lstm_X_test is not None and len(lstm_X_test) > 0:
                                    lstm_pred = lstm_model.predict(lstm_X_test, verbose=0)
                                    lstm_score = r2_score(lstm_y_test, lstm_pred)
                                    model_scores['lstm'] = max(0, lstm_score)
                                    self.models['lstm'] = lstm_model
                                    print(f"   ✅ LSTM R² Score: {lstm_score:.4f}")
                except Exception as e:
                    print(f"   ❌ LSTM training failed: {e}")
            
            # === 2. XGBOOST MODEL ===
            if XGBOOST_AVAILABLE:
                print("   📊 Training XGBoost model...")
                try:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    
                    xgb_model.fit(X_train_scaled, y_train)
                    xgb_pred = xgb_model.predict(X_test_scaled)
                    xgb_score = r2_score(y_test, xgb_pred)
                    
                    model_scores['xgboost'] = max(0, xgb_score)
                    self.models['xgboost'] = xgb_model
                    print(f"   ✅ XGBoost R² Score: {xgb_score:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ XGBoost training failed: {e}")
            
            # === 3. TRADITIONAL ML MODELS ===
            traditional_models = {
                'random_forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                'ridge': Ridge(alpha=1.0, random_state=42)
            }
            
            for name, model in traditional_models.items():
                print(f"   📊 Training {name}...")
                try:
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_test_scaled)
                    score = r2_score(y_test, pred)
                    
                    model_scores[name] = max(0, score)
                    self.models[name] = model
                    print(f"   ✅ {name} R² Score: {score:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ {name} training failed: {e}")
            
            # === 4. GARCH COMPONENT ===
            self.fit_garch_component(returns)
            
            # === 5. ENSEMBLE WEIGHTING ===
            if model_scores:
                # Use performance-based weighting with minimum threshold
                total_score = sum(max(0.01, score) for score in model_scores.values())
                self.ensemble_weights = {name: max(0.01, score)/total_score for name, score in model_scores.items()}
                
                self.is_trained = True
                
                best_score = max(model_scores.values())
                print(f"\n   🎯 Ensemble trained with {len(self.models)} models")
                print(f"   📈 Best individual score: {best_score:.4f}")
                print(f"   ⚖️ Ensemble weights: {dict(sorted(self.ensemble_weights.items(), key=lambda x: x[1], reverse=True))}")
                
                return best_score > 0.1  # Higher threshold for acceptance
            
            return False
            
        except Exception as e:
            print(f"Advanced ensemble training failed: {e}")
            return False
    
    def advanced_ensemble_forecast(self, price_data, days_ahead=30):
        """Generate sophisticated ensemble forecast"""
        try:
            if not self.is_trained:
                return 0.25
            
            # Create features for latest data
            features_df = self.create_advanced_features(price_data)
            if features_df is None:
                return 0.25
            
            # Get latest feature vector
            latest_features = features_df.iloc[-1:].fillna(0)
            latest_features_scaled = self.feature_scaler.transform(latest_features)
            
            # Get predictions from all models
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                try:
                    weight = self.ensemble_weights.get(name, 0)
                    if weight <= 0:
                        continue
                    
                    if name == 'lstm' and TENSORFLOW_AVAILABLE:
                        # LSTM requires sequence data
                        sequence_length = 20
                        if len(features_df) >= sequence_length:
                            sequence_data = features_df.iloc[-sequence_length:].fillna(0).values
                            sequence_data = sequence_data.reshape(1, sequence_length, -1)
                            pred = model.predict(sequence_data, verbose=0)[0][0]
                            predictions.append(pred)
                            weights.append(weight)
                    else:
                        # Traditional ML models
                        pred = model.predict(latest_features_scaled)[0]
                        predictions.append(pred)
                        weights.append(weight)
                        
                except Exception as e:
                    continue
            
            if predictions:
                # Weighted ensemble prediction
                ensemble_pred = np.average(predictions, weights=weights)
                
                # Apply GARCH adjustment if available
                if self.garch_params:
                    returns = price_data['Close'].pct_change().dropna()
                    garch_adjustment = self._get_garch_adjustment(returns)
                    ensemble_pred = ensemble_pred * garch_adjustment
                
                # Sanity bounds and ensure reasonable values
                ensemble_pred = max(0.05, min(1.0, ensemble_pred))
                
                return ensemble_pred
            
            # Fallback
            return 0.25
            
        except Exception as e:
            print(f"Advanced forecast error: {e}")
            return 0.25
    
    def _get_garch_adjustment(self, returns):
        """Get GARCH-based volatility adjustment"""
        try:
            if not self.garch_params:
                return 1.0
            
            # Simple GARCH adjustment based on recent volatility
            recent_vol = returns.tail(5).std() * np.sqrt(252)
            long_term_vol = returns.std() * np.sqrt(252)
            
            if recent_vol > 0 and long_term_vol > 0:
                vol_ratio = recent_vol / long_term_vol
                # Bounded adjustment
                adjustment = 0.8 + 0.4 * np.tanh(vol_ratio - 1)
                return max(0.5, min(2.0, adjustment))
            
            return 1.0
            
        except:
            return 1.0
    
    # Legacy interface methods
    def train_enhanced_model(self, price_data):
        """Legacy interface - use advanced ensemble"""
        return self.train_advanced_ensemble(price_data)
    
    def forecast_volatility(self, price_data, days_ahead=30):
        """Legacy interface - use advanced forecast"""
        return self.advanced_ensemble_forecast(price_data, days_ahead)


# Update the main class to use advanced version
class EnhancedVolatilityForecaster(AdvancedVolatilityForecaster):
    def __init__(self):
        super().__init__()

# Update the old VolatilityForecaster to use enhanced version
class VolatilityForecaster(EnhancedVolatilityForecaster):
    def __init__(self):
        super().__init__()
    
    def train_volatility_model(self, price_data):
        return self.train_enhanced_model(price_data)

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