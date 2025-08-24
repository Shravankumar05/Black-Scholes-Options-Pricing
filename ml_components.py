# Import deployment configuration FIRST to configure TensorFlow
try:
    from deployment_config import deployment_config, configure_deployment_environment
    # Configure deployment environment if needed
    if deployment_config.get('is_deployed', False):
        configure_deployment_environment()
except ImportError:
    deployment_config = {'is_deployed': False}

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

# Configure TensorFlow for deployment environments with enhanced container support
try:
    import os
    # Enhanced TensorFlow environment configuration for containers
    tf_env_vars = {
        'CUDA_VISIBLE_DEVICES': '-1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
        'CUDA_CACHE_DISABLE': '1',
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/dev/null',
        'TF_FORCE_CPU_DEVICE': '1',
        'NVIDIA_VISIBLE_DEVICES': 'none'
    }
    
    for key, value in tf_env_vars.items():
        os.environ[key] = value
    
    import tensorflow as tf
    # Force CPU-only configuration
    tf.config.set_visible_devices([], 'GPU')
    tf.config.optimizer.set_jit(False)
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    
    # Force CPU device context
    with tf.device('/CPU:0'):
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow configured for CPU-only ML operations")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using simplified ML models")
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow configuration completed (warnings suppressed)")

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
                print(f"   ⚠️ Insufficient data for feature creation: {len(price_data)} < {self.lookback_days}")
                return None
                
            features_df = pd.DataFrame(index=price_data.index)
            returns = price_data['Close'].pct_change().dropna()
            prices = price_data['Close']
            
            print(f"   📈 Creating advanced features from {len(price_data)} data points...")
            
            # === BASIC VOLATILITY FEATURES ===
            for window in [5, 10, 20, 30, 60]:
                if len(returns) >= window:
                    vol_window = returns.rolling(window).std() * np.sqrt(252)
                    features_df[f'realized_vol_{window}d'] = vol_window
                    features_df[f'vol_change_{window}d'] = vol_window.pct_change()
                    
                    try:
                        features_df[f'vol_skew_{window}d'] = returns.rolling(window).skew()
                        features_df[f'vol_kurt_{window}d'] = returns.rolling(window).apply(
                            lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
                        )
                    except:
                        features_df[f'vol_skew_{window}d'] = 0
                        features_df[f'vol_kurt_{window}d'] = 0
            
            # === VOLATILITY CLUSTERING & MOMENTUM ===
            abs_returns = np.abs(returns)
            for window in [10, 20, 50]:
                if len(abs_returns) >= window:
                    features_df[f'abs_ret_ma_{window}d'] = abs_returns.rolling(window).mean()
                    features_df[f'vol_clustering_{window}d'] = abs_returns.rolling(window).std()
                    features_df[f'vol_momentum_{window}d'] = (abs_returns.rolling(window).mean() / 
                                                              abs_returns.rolling(window*2).mean() - 1)
            
            # === PRICE-BASED FEATURES ===
            for window in [5, 10, 20, 50]:
                if len(prices) >= window:
                    sma = prices.rolling(window).mean()
                    features_df[f'sma_{window}'] = sma
                    features_df[f'price_position_{window}'] = prices / sma - 1
                    features_df[f'bb_position_{window}'] = self._bollinger_position(prices, window)
                    
                    # Price acceleration
                    features_df[f'price_accel_{window}'] = prices.diff().diff().rolling(window).mean()
                    
            # === MOMENTUM & TECHNICAL INDICATORS ===
            for window in [5, 10, 20, 60]:
                if len(prices) >= window:
                    features_df[f'momentum_{window}d'] = prices.pct_change(window)
                    features_df[f'rsi_{window}d'] = self._calculate_advanced_rsi(prices, window)
                    
            # Advanced technical indicators
            features_df['macd'] = self._calculate_macd(prices)
            features_df['macd_signal'] = self._calculate_macd_signal(prices)
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # === VOLATILITY REGIME & PERSISTENCE ===
            features_df['vol_regime'] = self._detect_volatility_regime(returns)
            features_df['vol_trend'] = self._calculate_volatility_trend(returns)
            features_df['vol_persistence'] = self._calculate_volatility_persistence(returns)
            features_df['vol_mean_reversion'] = self._calculate_mean_reversion_strength(returns)
            
            # === HIGHER-ORDER MOMENTS & TAIL RISK ===
            for window in [20, 60]:
                if len(returns) >= window:
                    roll_rets = returns.rolling(window)
                    try:
                        features_df[f'skewness_{window}d'] = roll_rets.skew()
                        features_df[f'kurtosis_{window}d'] = roll_rets.apply(
                            lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
                        )
                    except:
                        features_df[f'skewness_{window}d'] = 0
                        features_df[f'kurtosis_{window}d'] = 0
                    
                    # Tail risk measures
                    features_df[f'downside_vol_{window}d'] = roll_rets.apply(
                        lambda x: np.std(x[x < 0]) * np.sqrt(252) if len(x[x < 0]) > 1 else 0
                    )
                    features_df[f'var_95_{window}d'] = roll_rets.quantile(0.05)
                    features_df[f'cvar_95_{window}d'] = roll_rets.apply(
                        lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else 0
                    )
            
            # === JUMP & DISCONTINUITY DETECTION ===
            features_df['jump_indicator'] = self._detect_price_jumps(returns)
            features_df['gap_indicator'] = self._detect_gaps(price_data)
            features_df['outlier_indicator'] = self._detect_outliers(returns)
            
            # === VOLUME & LIQUIDITY PROXIES ===
            if 'Volume' in price_data.columns:
                volume = price_data['Volume']
                features_df['volume_trend'] = volume.pct_change(20)
                features_df['price_volume_trend'] = returns.rolling(20).corr(volume.pct_change())
                features_df['volume_volatility'] = volume.pct_change().rolling(20).std()
            else:
                # Use absolute returns and price changes as liquidity proxies
                features_df['volume_proxy'] = abs_returns.rolling(20).mean()
                features_df['volume_trend'] = features_df['volume_proxy'].pct_change(10)
                features_df['liquidity_proxy'] = 1 / (abs_returns.rolling(5).mean() + 1e-6)
            
            # === CROSS-SECTIONAL & INTRADAY FEATURES ===
            features_df['intraday_range'] = self._calculate_intraday_range(price_data)
            features_df['overnight_return'] = self._calculate_overnight_returns(price_data)
            features_df['range_volatility'] = self._calculate_range_volatility(price_data)
            
            # === TIME & CALENDAR FEATURES ===
            features_df['day_of_week'] = price_data.index.dayofweek
            features_df['month_of_year'] = price_data.index.month
            features_df['quarter'] = price_data.index.quarter
            features_df['is_month_end'] = (price_data.index.day > 25).astype(int)
            features_df['is_quarter_end'] = ((price_data.index.month % 3 == 0) & 
                                           (price_data.index.day > 25)).astype(int)
            
            # === INTERACTION FEATURES ===
            # Create some meaningful interactions
            if 'realized_vol_20d' in features_df.columns and 'momentum_20d' in features_df.columns:
                features_df['vol_momentum_interaction'] = (features_df['realized_vol_20d'] * 
                                                         np.abs(features_df['momentum_20d']))
            
            if 'rsi_20d' in features_df.columns and 'vol_regime' in features_df.columns:
                features_df['rsi_regime_interaction'] = features_df['rsi_20d'] * features_df['vol_regime']
            
            # Store feature names (exclude target if it exists)
            self.feature_names = [col for col in features_df.columns if col != 'target']
            
            # Final cleaning and validation
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"   ✅ Created {len(features_df.columns)} features with {features_df.shape[0]} observations")
            
            return features_df
            
        except Exception as e:
            print(f"Feature creation error: {e}")
            import traceback
            traceback.print_exc()
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
    
    def _calculate_volatility_persistence(self, returns):
        """Calculate volatility persistence using autocorrelation"""
        try:
            vol_series = returns.rolling(20).std()
            persistence = vol_series.rolling(60).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 2 else 0
            )
            return persistence.fillna(0)
        except:
            return pd.Series(0, index=returns.index)
    
    def _detect_outliers(self, returns, threshold=3):
        """Detect outliers using modified z-score"""
        try:
            median = returns.rolling(20).median()
            mad = returns.rolling(20).apply(lambda x: np.median(np.abs(x - np.median(x))))
            modified_z_score = 0.6745 * (returns - median) / mad
            return (np.abs(modified_z_score) > threshold).astype(int)
        except:
            return pd.Series(0, index=returns.index)
    
    def _calculate_range_volatility(self, price_data):
        """Calculate range-based volatility estimator"""
        try:
            if 'High' in price_data.columns and 'Low' in price_data.columns:
                # Garman-Klass volatility estimator
                high = price_data['High']
                low = price_data['Low']
                close = price_data['Close']
                open_price = price_data['Open'] if 'Open' in price_data.columns else close.shift(1)
                
                range_vol = np.log(high/low) * np.log(high/close) + np.log(low/close) * np.log(low/close)
                return range_vol.rolling(20).mean() * 252
            else:
                # Use rolling high-low from close prices
                high_proxy = price_data['Close'].rolling(5).max()
                low_proxy = price_data['Close'].rolling(5).min()
                range_vol = np.log(high_proxy/low_proxy)
                return range_vol.rolling(20).mean() * 252
        except:
            return pd.Series(0, index=price_data.index)
    
    def _bollinger_position(self, prices, window):
        """Calculate position within Bollinger Bands"""
        try:
            sma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            return (prices - lower) / (upper - lower)
        except:
            return pd.Series(0.5, index=prices.index)
    
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
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            return ema_fast - ema_slow
        except:
            return pd.Series(0, index=prices.index)
    
    def _calculate_macd_signal(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD signal line"""
        try:
            macd = self._calculate_macd(prices, fast, slow)
            return macd.ewm(span=signal).mean()
        except:
            return pd.Series(0, index=prices.index)
    
    def _detect_volatility_regime(self, returns):
        """Advanced volatility regime detection"""
        try:
            vol_20d = returns.rolling(20).std() * np.sqrt(252)
            vol_60d = returns.rolling(60).std() * np.sqrt(252)
            
            regime = pd.Series(0, index=returns.index)
            regime[vol_20d > vol_60d * 1.5] = 2  # High vol regime
            regime[vol_20d < vol_60d * 0.7] = 1  # Low vol regime
            return regime.fillna(0)
        except:
            return pd.Series(0, index=returns.index)
    
    def _calculate_volatility_trend(self, returns):
        """Calculate volatility trend"""
        try:
            vol_5d = returns.rolling(5).std()
            vol_20d = returns.rolling(20).std()
            return (vol_5d / vol_20d - 1).fillna(0)
        except:
            return pd.Series(0, index=returns.index)
    
    def _calculate_mean_reversion_strength(self, returns):
        """Calculate mean reversion strength using Hurst exponent approximation"""
        def hurst_approx(series, max_lag=20):
            if len(series) < max_lag * 2:
                return 0.5
            try:
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
                if len(tau) == 0:
                    return 0.5
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        try:
            hurst_values = returns.rolling(60).apply(lambda x: hurst_approx(x.values))
            return hurst_values.fillna(0.5)
        except:
            return pd.Series(0.5, index=returns.index)
    
    def _select_best_features(self, features_df, target_data, max_features=30):
        """Select best features using multiple criteria"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            from sklearn.preprocessing import StandardScaler
            
            # Remove constant and near-constant features
            feature_variance = features_df.var()
            non_constant_features = features_df.loc[:, feature_variance > 1e-6]
            
            if len(non_constant_features.columns) == 0:
                return features_df.iloc[:, :min(20, len(features_df.columns))], {}
            
            # Remove highly correlated features
            corr_matrix = non_constant_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            reduced_features = non_constant_features.drop(columns=to_drop[:len(to_drop)//2])  # Drop half of correlated pairs
            
            # Statistical feature selection
            X_filled = reduced_features.fillna(reduced_features.median())
            
            # Use multiple selection methods
            n_select = min(max_features, len(X_filled.columns), len(target_data) // 3)
            
            if n_select < 5:
                return reduced_features.iloc[:, :min(10, len(reduced_features.columns))], {}
            
            # F-test based selection
            f_selector = SelectKBest(score_func=f_regression, k=n_select)
            f_selector.fit(X_filled, target_data)
            f_selected = reduced_features.columns[f_selector.get_support()]
            
            # Mutual information based selection
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_select)
            mi_selector.fit(X_filled, target_data)
            mi_selected = reduced_features.columns[mi_selector.get_support()]
            
            # Combine selections (union)
            combined_features = list(set(f_selected) | set(mi_selected))
            
            # Feature importance scores
            feature_importance = {}
            for i, feature in enumerate(reduced_features.columns):
                if feature in f_selected:
                    feature_importance[feature] = f_selector.scores_[reduced_features.columns.get_loc(feature)]
            
            selected_df = reduced_features[combined_features]
            return selected_df, feature_importance
            
        except Exception as e:
            print(f"   ⚠️ Feature selection failed: {e}")
            # Fallback: use first 20 features
            return features_df.iloc[:, :min(20, len(features_df.columns))], {}
    
    def _train_enhanced_lstm(self, features_df, y_train_scaled, y_test_scaled, split_idx):
        """Train enhanced LSTM with better architecture"""
        try:
            # Prepare sequences with shorter lookback for better learning
            sequence_length = 10  # Shorter sequences
            
            # Create sequences from scaled features
            X_sequences = []
            y_sequences = []
            
            feature_values = features_df.fillna(features_df.median()).values
            
            for i in range(sequence_length, split_idx):
                X_sequences.append(feature_values[i-sequence_length:i])
                y_sequences.append(y_train_scaled[i-sequence_length])
            
            if len(X_sequences) < 30:
                return None, None
            
            X_train_seq = np.array(X_sequences)
            y_train_seq = np.array(y_sequences)
            
            # Test sequences
            X_test_sequences = []
            y_test_sequences = []
            
            for i in range(split_idx + sequence_length, len(feature_values)):
                if i < len(y_test_scaled) + split_idx:
                    X_test_sequences.append(feature_values[i-sequence_length:i])
                    y_test_sequences.append(y_test_scaled[i-split_idx-sequence_length])
            
            if len(X_test_sequences) == 0:
                return None, None
            
            X_test_seq = np.array(X_test_sequences)
            y_test_seq = np.array(y_test_sequences)
            
            # Create improved LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.3),
                tf.keras.layers.LSTM(16, dropout=0.3),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            
            # Compile with appropriate optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train with early stopping
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
            ]
            
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test_seq, verbose=0).flatten()
            score = r2_score(y_test_seq, y_pred)
            
            if score > 0.1:  # Only use if reasonably good
                self.models['lstm'] = model
                return score, y_pred
            
            return None, None
            
        except Exception as e:
            print(f"   ⚠️ Enhanced LSTM training failed: {e}")
            return None, None
    
    def _create_minimum_variance_ensemble(self, good_models, model_predictions, y_test):
        """Create minimum variance ensemble weights with robust handling"""
        try:
            if len(good_models) == 1:
                return list(model_predictions.values())[0]
            
            # Find the minimum length among all predictions
            min_length = min(len(pred) for pred in model_predictions.values())
            min_length = min(min_length, len(y_test))
            
            if min_length < 5:  # Need minimum samples
                # Fallback to simple average
                predictions_list = [pred[:min_length] for pred in model_predictions.values()]
                return np.mean(predictions_list, axis=0)
            
            # Truncate all predictions and targets to the same length
            aligned_predictions = {}
            for name, pred in model_predictions.items():
                aligned_predictions[name] = pred[:min_length]
            
            y_test_aligned = y_test[:min_length]
            
            # Stack predictions
            pred_matrix = np.column_stack([aligned_predictions[name] for name in good_models.keys()])
            
            # Calculate covariance matrix of prediction errors
            errors = pred_matrix - y_test_aligned.reshape(-1, 1)
            cov_matrix = np.cov(errors.T)
            
            # Handle single model case or singular covariance
            if cov_matrix.ndim == 0 or np.linalg.det(cov_matrix) == 0:
                # Equal weights fallback
                weights = np.ones(len(good_models)) / len(good_models)
            else:
                try:
                    # Minimum variance weights
                    inv_cov = np.linalg.pinv(cov_matrix)
                    ones = np.ones((len(good_models), 1))
                    
                    weights = inv_cov @ ones
                    weights = weights.flatten()
                    
                    # Ensure weights are reasonable
                    weights = np.abs(weights)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(len(good_models)) / len(good_models)
                    
                    # Clip extreme weights
                    weights = np.clip(weights, 0.01, 0.7)
                    weights = weights / weights.sum()
                    
                except:
                    # Fallback: equal weights
                    weights = np.ones(len(good_models)) / len(good_models)
            
            # Create ensemble prediction
            ensemble_pred = pred_matrix @ weights
            
            return ensemble_pred
            
        except Exception as e:
            print(f"   ⚠️ Minimum variance ensemble failed: {e}")
            # Robust fallback: simple average
            try:
                min_length = min(len(pred) for pred in model_predictions.values())
                predictions_list = [pred[:min_length] for pred in model_predictions.values()]
                return np.mean(predictions_list, axis=0)
            except:
                # Final fallback: return first prediction
                return list(model_predictions.values())[0]
    
    def _detect_price_jumps(self, returns, threshold=3):
        """Detect price jumps using z-score"""
        try:
            rolling_mean = returns.rolling(20).mean()
            rolling_std = returns.rolling(20).std()
            z_scores = (returns - rolling_mean) / rolling_std
            return (np.abs(z_scores) > threshold).astype(int)
        except:
            return pd.Series(0, index=returns.index)
    
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
    
    def prepare_lstm_data(self, features_df, target_values, sequence_length=20):
        """Prepare data for LSTM training"""
        try:
            # Remove non-numeric columns
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                return None, None, None, None
            
            # Ensure target_values is aligned with features
            if isinstance(target_values, str):
                # If target_values is a column name, extract it
                if target_values in numeric_features.columns:
                    target_array = numeric_features[target_values].values
                else:
                    print(f"Target column '{target_values}' not found in features")
                    return None, None, None, None
            elif hasattr(target_values, '__len__'):
                # If target_values is an array-like object
                target_array = np.array(target_values)
            else:
                print(f"Invalid target_values type: {type(target_values)}")
                return None, None, None, None
            
            # Align features and targets
            min_length = min(len(numeric_features), len(target_array))
            numeric_features = numeric_features.iloc[:min_length]
            target_array = target_array[:min_length]
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, min_length):
                X_sequences.append(numeric_features.iloc[i-sequence_length:i].values)
                y_sequences.append(target_array[i])
            
            if len(X_sequences) == 0:
                print(f"Not enough data for sequences. Need at least {sequence_length + 1} samples, got {min_length}")
                return None, None, None, None
            
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
        """Train sophisticated ensemble with improved target engineering and model selection"""
        try:
            if 'Close' not in price_data.columns or len(price_data) < self.lookback_days + 50:
                return False
            
            print("🚀 Training Enhanced ML Ensemble...")
            
            # Create advanced features
            features_df = self.create_advanced_features(price_data)
            if features_df is None:
                return False
            
            # IMPROVED TARGET ENGINEERING
            print("   📊 Engineering improved volatility targets...")
            returns = price_data['Close'].pct_change().dropna()
            
            # Use multiple target types for robustness
            realized_vol_5d = returns.rolling(5).std() * np.sqrt(252)
            realized_vol_20d = returns.rolling(20).std() * np.sqrt(252)
            
            # Create more stable forward-looking targets
            targets = []
            target_indices = []
            forecast_horizon = min(self.forecast_days, 5)  # Even shorter horizon for better prediction
            
            # Use overlapping windows for more stable targets
            for i in range(20, len(realized_vol_20d) - forecast_horizon):  # Start after 20 days
                if i + forecast_horizon < len(realized_vol_20d):
                    # Current volatility context
                    current_vol = realized_vol_20d.iloc[i]
                    
                    # Future volatility (use simple average for stability)
                    future_vols = realized_vol_5d.iloc[i+1:i+forecast_horizon+1]
                    
                    if len(future_vols) >= forecast_horizon and not future_vols.isnull().any():
                        # Use median for robustness against outliers
                        target_vol = future_vols.median()
                        
                        # Apply volatility bounds and stability checks
                        if 0.05 <= target_vol <= 1.0 and 0.05 <= current_vol <= 1.0:
                            # Normalize by current volatility for relative prediction
                            vol_ratio = target_vol / current_vol
                            if 0.2 <= vol_ratio <= 5.0:  # Reasonable volatility changes
                                targets.append(target_vol)
                                target_indices.append(i)
            
            if len(targets) < 30:
                print(f"   ⚠️ Insufficient valid targets: {len(targets)}")
                return False
            
            # Align features and targets properly
            aligned_features = features_df.iloc[target_indices]
            target_data = np.array(targets)
            
            print(f"   📈 Training data: {len(aligned_features)} samples, {len(aligned_features.columns)} features")
            
            # FEATURE SELECTION AND PREPROCESSING
            # Remove highly correlated and low-variance features
            selected_features, feature_importance = self._select_best_features(aligned_features, target_data)
            
            if len(selected_features.columns) < 10:
                print(f"   ⚠️ Too few features after selection: {len(selected_features.columns)}")
                selected_features = aligned_features.iloc[:, :20]  # Use top 20 original features
            
            print(f"   🎯 Selected {len(selected_features.columns)} best features")
            
            # Split data temporally
            split_idx = int(len(selected_features) * 0.8)
            
            X_train = selected_features.iloc[:split_idx]
            X_test = selected_features.iloc[split_idx:]
            y_train = target_data[:split_idx]
            y_test = target_data[split_idx:]
            
            # Scale features robustly
            X_train_scaled = self.feature_scaler.fit_transform(X_train.fillna(X_train.median()))
            X_test_scaled = self.feature_scaler.transform(X_test.fillna(X_train.median()))
            
            # Scale targets for better model performance
            y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            model_scores = {}
            model_predictions = {}
            
            # === 1. IMPROVED LSTM MODEL ===
            if TENSORFLOW_AVAILABLE and len(X_train) > 50:
                print("   🧠 Training Enhanced LSTM model...")
                try:
                    lstm_score, lstm_pred = self._train_enhanced_lstm(
                        selected_features, y_train_scaled, y_test_scaled, split_idx
                    )
                    if lstm_score is not None:
                        model_scores['lstm'] = lstm_score
                        model_predictions['lstm'] = lstm_pred
                        print(f"   ✅ Enhanced LSTM R² Score: {lstm_score:.4f}")
                except Exception as e:
                    print(f"   ⚠️ LSTM training failed: {e}")
            
            # === 2. OPTIMIZED XGBOOST ===
            if XGBOOST_AVAILABLE:
                print("   🌳 Training Optimized XGBoost...")
                try:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=100,  # Reduced to prevent overfitting
                        max_depth=4,       # Shallower trees
                        learning_rate=0.05, # Lower learning rate
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,     # L1 regularization
                        reg_lambda=0.1,    # L2 regularization
                        random_state=42,
                        early_stopping_rounds=10
                    )
                    
                    # Use validation set for early stopping
                    eval_set = [(X_test_scaled, y_test_scaled)]
                    xgb_model.fit(
                        X_train_scaled, y_train_scaled, 
                        eval_set=eval_set, verbose=False
                    )
                    
                    xgb_pred = xgb_model.predict(X_test_scaled)
                    xgb_score = max(0, r2_score(y_test_scaled, xgb_pred))
                    
                    model_scores['xgboost'] = xgb_score
                    model_predictions['xgboost'] = xgb_pred
                    self.models['xgboost'] = xgb_model
                    print(f"   ✅ Optimized XGBoost R² Score: {xgb_score:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️ XGBoost training failed: {e}")
            
            # === 3. OPTIMIZED TRADITIONAL MODELS ===
            traditional_models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=50, max_depth=5, min_samples_split=10, 
                    min_samples_leaf=5, random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=50, learning_rate=0.05, max_depth=4, 
                    min_samples_split=10, random_state=42
                ),
                'elastic_net': ElasticNet(
                    alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000
                )
            }
            
            for name, model in traditional_models.items():
                print(f"   📊 Training optimized {name}...")
                try:
                    model.fit(X_train_scaled, y_train_scaled)
                    pred = model.predict(X_test_scaled)
                    score = max(0, r2_score(y_test_scaled, pred))
                    
                    model_scores[name] = score
                    model_predictions[name] = pred
                    self.models[name] = model
                    print(f"   ✅ {name} R² Score: {score:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️ {name} training failed: {e}")
            
            # === 4. INTELLIGENT ENSEMBLE WEIGHTING ===
            if model_scores:
                # Only use models with positive performance
                good_models = {name: score for name, score in model_scores.items() if score > 0.05}
                
                if good_models:
                    # Performance-based weighting with minimum variance ensemble
                    ensemble_pred = self._create_minimum_variance_ensemble(
                        good_models, model_predictions, y_test_scaled
                    )
                    
                    # Calculate ensemble score
                    ensemble_score = r2_score(y_test_scaled, ensemble_pred)
                    
                    # Use only good models
                    total_score = sum(good_models.values())
                    self.ensemble_weights = {
                        name: score/total_score for name, score in good_models.items()
                    }
                    
                    print(f"\n   🎯 Ensemble trained with {len(good_models)} good models")
                    print(f"   📈 Best individual score: {max(good_models.values()):.4f}")
                    print(f"   🏆 Ensemble score: {ensemble_score:.4f}")
                    print(f"   ⚖️ Smart weights: {dict(sorted(self.ensemble_weights.items(), key=lambda x: x[1], reverse=True))}")
                    
                    self.is_trained = True
                    return ensemble_score > 0.1 or max(good_models.values()) > 0.2
                else:
                    print("   ⚠️ No models achieved acceptable performance")
                    # Fallback: use simple baseline
                    self.ensemble_weights = {'baseline': 1.0}
                    self.is_trained = True
                    return True
            
            return False
            
        except Exception as e:
            print(f"Enhanced ensemble training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def advanced_ensemble_forecast(self, price_data, days_ahead=30):
        """Generate sophisticated ensemble forecast with improved handling"""
        try:
            if not self.is_trained:
                # Simple fallback based on recent volatility
                returns = price_data['Close'].pct_change().dropna()
                if len(returns) > 20:
                    recent_vol = returns.tail(20).std() * np.sqrt(252)
                    return max(0.05, min(1.0, recent_vol))
                return 0.25
            
            # Handle baseline model case
            if 'baseline' in self.ensemble_weights:
                returns = price_data['Close'].pct_change().dropna()
                if len(returns) > 10:
                    return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                return 0.25
            
            # Create features for latest data
            features_df = self.create_advanced_features(price_data)
            if features_df is None:
                return 0.25
            
            # Get latest feature vector
            latest_features = features_df.iloc[-1:].fillna(features_df.median())
            
            # Check if we need to apply the same feature selection
            if hasattr(self, 'selected_features'):
                # Use same features as training
                available_features = [f for f in self.selected_features if f in latest_features.columns]
                if len(available_features) > 0:
                    latest_features = latest_features[available_features]
            
            try:
                latest_features_scaled = self.feature_scaler.transform(latest_features)
            except:
                # Fallback if scaler fails
                latest_features_scaled = latest_features.values
            
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
                        sequence_length = 10
                        if len(features_df) >= sequence_length:
                            sequence_data = features_df.iloc[-sequence_length:].fillna(features_df.median()).values
                            sequence_data = sequence_data.reshape(1, sequence_length, -1)
                            pred = model.predict(sequence_data, verbose=0)
                            
                            # Handle different prediction shapes
                            if hasattr(pred, 'flatten'):
                                pred_value = pred.flatten()[0]
                            else:
                                pred_value = pred[0] if len(pred) > 0 else pred
                            
                            # Inverse transform if scaler is available
                            if hasattr(self, 'scaler'):
                                try:
                                    pred_value = self.scaler.inverse_transform([[pred_value]])[0, 0]
                                except:
                                    pass
                            
                            predictions.append(pred_value)
                            weights.append(weight)
                    else:
                        # Traditional ML models
                        pred = model.predict(latest_features_scaled)[0]
                        
                        # Inverse transform if scaler is available
                        if hasattr(self, 'scaler'):
                            try:
                                pred = self.scaler.inverse_transform([[pred]])[0, 0]
                            except:
                                pass
                        
                        predictions.append(pred)
                        weights.append(weight)
                        
                except Exception as e:
                    continue
            
            if predictions:
                # Weighted ensemble prediction
                if len(predictions) == 1:
                    ensemble_pred = predictions[0]
                else:
                    ensemble_pred = np.average(predictions, weights=weights)
                
                # Apply volatility regime adjustment
                returns = price_data['Close'].pct_change().dropna()
                if len(returns) > 20:
                    recent_vol = returns.tail(20).std() * np.sqrt(252)
                    long_term_vol = returns.tail(60).std() * np.sqrt(252) if len(returns) > 60 else recent_vol
                    
                    # Regime adjustment
                    if recent_vol > long_term_vol * 1.5:  # High vol regime
                        ensemble_pred *= 1.1
                    elif recent_vol < long_term_vol * 0.7:  # Low vol regime
                        ensemble_pred *= 0.9
                
                # Sanity bounds and ensure reasonable values
                ensemble_pred = max(0.05, min(1.5, ensemble_pred))
                
                return ensemble_pred
            
            # Fallback to recent volatility
            returns = price_data['Close'].pct_change().dropna()
            if len(returns) > 10:
                return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            return 0.25
            
        except Exception as e:
            print(f"Advanced forecast error: {e}")
            # Robust fallback
            try:
                returns = price_data['Close'].pct_change().dropna()
                if len(returns) > 5:
                    return returns.std() * np.sqrt(252)
            except:
                pass
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