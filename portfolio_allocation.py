import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from sklearn.covariance import LedoitWolf, EmpiricalCovariance, OAS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.mixture import GaussianMixture
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class AdvancedPortfolioAllocationEngine:
    def __init__(self):
        self.risk_tolerance_profiles = {
            'conservative': {'max_volatility': 0.12, 'target_return': 0.06},
            'moderate': {'max_volatility': 0.18, 'target_return': 0.09},
            'aggressive': {'max_volatility': 0.30, 'target_return': 0.15}
        }
        self.regime_model = None
        self.return_predictor = None
        self.risk_factor_model = None
        self.scaler = StandardScaler()
        
        # Advanced parameters
        self.transaction_cost = 0.001  # 10 bps
        self.rebalance_threshold = 0.05
        self.confidence_level = 0.95
        
    def create_market_features(self, returns_data):
        """Create advanced market features for regime detection and return prediction"""
        try:
            features_df = pd.DataFrame(index=returns_data.index)
            
            # === VOLATILITY FEATURES ===
            for window in [5, 10, 20, 60]:
                vol = returns_data.rolling(window).std() * np.sqrt(252)
                features_df[f'avg_vol_{window}d'] = vol.mean(axis=1)
                features_df[f'vol_dispersion_{window}d'] = vol.std(axis=1)
            
            # === CORRELATION FEATURES ===
            for window in [20, 60]:
                corr_matrix = returns_data.rolling(window).corr()
                # Average correlation (excluding diagonal)
                avg_corr = []
                for date in returns_data.index[window:]:
                    if date in corr_matrix.index:
                        corr_subset = corr_matrix.loc[date]
                        if len(corr_subset.shape) == 2:
                            corr_values = corr_subset.values
                            mask = ~np.eye(corr_values.shape[0], dtype=bool)
                            avg_corr.append(np.nanmean(corr_values[mask]))
                        else:
                            avg_corr.append(0)
                    else:
                        avg_corr.append(0)
                
                features_df.loc[returns_data.index[window:], f'avg_correlation_{window}d'] = avg_corr
            
            # === MOMENTUM FEATURES ===
            for window in [5, 20, 60]:
                momentum = returns_data.rolling(window).mean() * 252
                features_df[f'avg_momentum_{window}d'] = momentum.mean(axis=1)
                features_df[f'momentum_dispersion_{window}d'] = momentum.std(axis=1)
            
            # === MARKET STRESS INDICATORS ===
            # Drawdown indicators
            cumulative_returns = (1 + returns_data).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            features_df['max_drawdown'] = drawdowns.min(axis=1)
            features_df['avg_drawdown'] = drawdowns.mean(axis=1)
            
            # Tail risk measures
            for window in [20, 60]:
                returns_window = returns_data.rolling(window)
                features_df[f'var_95_{window}d'] = returns_window.quantile(0.05).mean(axis=1)
                try:
                    features_df[f'skewness_{window}d'] = returns_window.skew().mean(axis=1)
                    # Use apply for kurtosis compatibility
                    kurt_values = []
                    for date in returns_data.index[window:]:
                        window_data = returns_data.loc[:date].tail(window)
                        kurt_val = window_data.apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0).mean()
                        kurt_values.append(kurt_val)
                    features_df.loc[returns_data.index[window:], f'kurtosis_{window}d'] = kurt_values
                except:
                    features_df[f'skewness_{window}d'] = 0
                    features_df[f'kurtosis_{window}d'] = 0
            
            return features_df.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            print(f"Feature creation error: {e}")
            return pd.DataFrame(index=returns_data.index)
    
    def detect_market_regime_hmm(self, returns_data):
        """Advanced regime detection using Hidden Markov Model"""
        try:
            if not HMM_AVAILABLE or len(returns_data) < 100:
                return self.detect_market_regime_simple(returns_data)
            
            # Create features for regime detection
            features_df = self.create_market_features(returns_data)
            
            if len(features_df.columns) == 0:
                return self.detect_market_regime_simple(returns_data)
            
            # Use key features for regime detection
            regime_features = [
                'avg_vol_60d', 'avg_correlation_60d', 'avg_momentum_60d',
                'max_drawdown', 'var_95_60d'
            ]
            
            available_features = [f for f in regime_features if f in features_df.columns]
            if len(available_features) < 3:
                return self.detect_market_regime_simple(returns_data)
            
            feature_data = features_df[available_features].fillna(0)
            
            # Fit Gaussian Mixture Model (simplified HMM)
            n_regimes = 3  # Bull, Bear, Sideways
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            
            # Fit model
            scaled_features = self.scaler.fit_transform(feature_data)
            regime_labels = gmm.fit_predict(scaled_features)
            
            # Map regime labels to meaningful names
            regime_stats = {}
            for regime in range(n_regimes):
                mask = regime_labels == regime
                if mask.sum() > 0:
                    avg_return = returns_data[mask].mean().mean() * 252
                    avg_vol = returns_data[mask].std().mean() * np.sqrt(252)
                    regime_stats[regime] = {'return': avg_return, 'vol': avg_vol}
            
            # Classify regimes
            regime_names = {}
            sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['return'])
            
            if len(sorted_regimes) >= 3:
                regime_names[sorted_regimes[0][0]] = 'bear_market'
                regime_names[sorted_regimes[1][0]] = 'sideways'
                regime_names[sorted_regimes[2][0]] = 'bull_market'
            
            # Get current regime
            current_regime_id = regime_labels[-1]
            current_regime = regime_names.get(current_regime_id, 'normal')
            
            # Store model for future use
            self.regime_model = {'gmm': gmm, 'scaler': self.scaler, 'mapping': regime_names}
            
            return current_regime
            
        except Exception as e:
            print(f"HMM regime detection failed: {e}")
            return self.detect_market_regime_simple(returns_data)
    
    def detect_market_regime_simple(self, returns_data):
        """Simple fallback regime detection"""
        try:
            if len(returns_data) < 60:
                return 'insufficient_data'
            
            # Calculate indicators
            recent_returns = returns_data.tail(20)
            avg_return = recent_returns.mean().mean()
            avg_vol = recent_returns.std().mean()
            
            # Correlation analysis
            recent_corr = recent_returns.corr().values
            avg_correlation = (recent_corr.sum() - np.diag(recent_corr).sum()) / (recent_corr.size - len(recent_corr))
            
            # Regime classification
            if avg_vol > 0.025:  # High volatility (daily)
                if avg_correlation > 0.7:
                    return 'crisis'  # High vol + high correlation
                else:
                    return 'high_volatility'
            elif avg_return > 0.001:  # Positive returns
                return 'bull_market'
            elif avg_return < -0.001:  # Negative returns
                return 'bear_market'
            else:
                return 'sideways'
                
        except Exception as e:
            print(f"Simple regime detection error: {e}")
            return 'normal'
    def train_return_predictor(self, returns_data):
        """Train ML model to predict asset returns"""
        try:
            print("🤖 Training ML return predictor...")
            
            features_df = self.create_market_features(returns_data)
            if len(features_df.columns) == 0:
                return False
            
            # Create targets (forward returns)
            targets = {}
            for asset in returns_data.columns:
                asset_targets = []
                for i in range(len(returns_data) - 20):
                    future_return = returns_data[asset].iloc[i+1:i+21].mean()  # 20-day forward return
                    asset_targets.append(future_return)
                targets[asset] = asset_targets
            
            # Align features and targets
            min_length = min(len(features_df) - 20, min(len(t) for t in targets.values()))
            feature_data = features_df.iloc[:min_length]
            
            self.return_predictor = {}
            
            for asset in returns_data.columns:
                try:
                    target_data = np.array(targets[asset][:min_length])
                    
                    if len(target_data) < 50:
                        continue
                    
                    # Split data
                    split_idx = int(len(feature_data) * 0.8)
                    X_train = feature_data.iloc[:split_idx].fillna(0)
                    X_test = feature_data.iloc[split_idx:].fillna(0)
                    y_train = target_data[:split_idx]
                    y_test = target_data[split_idx:]
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Validate
                    pred = model.predict(X_test)
                    score = np.corrcoef(pred, y_test)[0,1] if len(pred) > 1 else 0
                    
                    if not np.isnan(score) and score > 0.05:  # Minimum correlation threshold
                        self.return_predictor[asset] = model
                        print(f"   ✅ {asset} predictor correlation: {score:.3f}")
                    
                except Exception as e:
                    continue
            
            return len(self.return_predictor) > 0
            
        except Exception as e:
            print(f"Return predictor training failed: {e}")
            return False
    
    def predict_expected_returns(self, returns_data):
        """Predict expected returns using ML models"""
        try:
            if not self.return_predictor:
                # Fallback to historical means
                return returns_data.mean() * 252
            
            features_df = self.create_market_features(returns_data)
            if len(features_df.columns) == 0:
                return returns_data.mean() * 252
            
            latest_features = features_df.iloc[-1:].fillna(0)
            predicted_returns = {}
            
            for asset, model in self.return_predictor.items():
                try:
                    pred = model.predict(latest_features)[0]
                    # Annualize and apply bounds
                    annual_pred = pred * 252
                    annual_pred = max(-0.5, min(0.5, annual_pred))  # Reasonable bounds
                    predicted_returns[asset] = annual_pred
                except:
                    # Fallback to historical mean
                    predicted_returns[asset] = returns_data[asset].mean() * 252
            
            # Fill in any missing assets
            for asset in returns_data.columns:
                if asset not in predicted_returns:
                    predicted_returns[asset] = returns_data[asset].mean() * 252
            
            return pd.Series(predicted_returns)
            
        except Exception as e:
            print(f"Return prediction error: {e}")
            return returns_data.mean() * 252
    
    def ml_enhanced_black_litterman(self, returns_data, assets, confidence_level=0.8):
        """ML-enhanced Black-Litterman allocation"""
        try:
            print("🧠 Running ML-Enhanced Black-Litterman...")
            
            # Train return predictor if not available
            if not self.return_predictor:
                self.train_return_predictor(returns_data)
            
            # Market capitalization weights (equal for simplicity)
            market_caps = {asset: 1.0 for asset in assets}
            total_market_cap = sum(market_caps.values())
            prior_weights = np.array([market_caps.get(asset, 0) / total_market_cap for asset in assets])
            
            # Robust covariance estimation
            cov_estimators = {
                'ledoit_wolf': LedoitWolf(),
                'oas': OAS(),
                'empirical': EmpiricalCovariance()
            }
            
            best_cov = None
            best_score = -np.inf
            
            for name, estimator in cov_estimators.items():
                try:
                    cov_matrix = estimator.fit(returns_data.fillna(0)).covariance_
                    # Score based on condition number (lower is better)
                    score = -np.linalg.cond(cov_matrix)
                    if score > best_score:
                        best_score = score
                        best_cov = cov_matrix
                except:
                    continue
            
            if best_cov is None:
                best_cov = returns_data.cov().values
            
            annual_cov = best_cov * 252
            
            # ML-predicted expected returns
            ml_returns = self.predict_expected_returns(returns_data)
            
            # Black-Litterman parameters
            risk_aversion = 3.0
            tau = 0.025  # Uncertainty parameter
            
            # Prior (equilibrium) returns
            prior_returns = risk_aversion * np.dot(annual_cov, prior_weights)
            
            # Views: Use ML predictions as views
            P = np.eye(len(assets))  # Pick matrix (views on all assets)
            Q = ml_returns.values  # ML predictions as views
            
            # Uncertainty in views (higher uncertainty for assets with poor ML predictions)
            omega_diag = []
            for asset in assets:
                if asset in self.return_predictor:
                    # Lower uncertainty for assets with good predictors
                    uncertainty = 0.1 * annual_cov[assets.index(asset), assets.index(asset)]
                else:
                    # Higher uncertainty for assets without good predictors
                    uncertainty = 0.5 * annual_cov[assets.index(asset), assets.index(asset)]
                omega_diag.append(uncertainty)
            
            Omega = np.diag(omega_diag)
            
            # Black-Litterman formula
            try:
                # Posterior covariance
                tau_cov = tau * annual_cov
                inv_tau_cov = np.linalg.inv(tau_cov)
                inv_omega = np.linalg.inv(Omega)
                
                posterior_cov_inv = inv_tau_cov + np.dot(P.T, np.dot(inv_omega, P))
                posterior_cov = np.linalg.inv(posterior_cov_inv)
                
                # Posterior returns
                posterior_returns = np.dot(posterior_cov, 
                                         np.dot(inv_tau_cov, prior_returns) + 
                                         np.dot(P.T, np.dot(inv_omega, Q)))
                
                # Optimize portfolio
                def portfolio_utility(weights):
                    weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
                    port_return = np.sum(weights * posterior_returns)
                    port_var = np.dot(weights.T, np.dot(posterior_cov, weights))
                    return -(port_return - 0.5 * risk_aversion * port_var)
                
                # Constraints and bounds
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
                bounds = tuple((0.001, 0.30) for _ in range(len(assets)))
                
                # Multiple optimization attempts
                best_result = None
                best_utility = -np.inf
                
                for seed in [42, 123, 456]:
                    np.random.seed(seed)
                    initial_guess = np.random.dirichlet(np.ones(len(assets)))
                    
                    try:
                        result = minimize(portfolio_utility, initial_guess, method='SLSQP',
                                        bounds=bounds, constraints=constraints,
                                        options={'maxiter': 1000})
                        
                        if result.success and -result.fun > best_utility:
                            best_result = result
                            best_utility = -result.fun
                    except:
                        continue
                
                if best_result and best_result.success:
                    weights = np.abs(best_result.x)
                    weights = weights / weights.sum()
                    
                    allocation = dict(zip(assets, weights))
                    print(f"   ✅ ML-BL optimization successful")
                    return allocation
                    
            except np.linalg.LinAlgError:
                print("   ⚠️ Matrix inversion failed, using simplified approach")
            
            # Fallback: Use ML predictions directly with mean-variance optimization
            return self.ml_mean_variance_optimization(returns_data, assets, ml_returns)
            
        except Exception as e:
            print(f"ML Black-Litterman failed: {e}")
            return self.equal_weight_allocation(assets)
    def ml_mean_variance_optimization(self, returns_data, assets, expected_returns):
        """Mean-variance optimization with ML-predicted returns"""
        try:
            # Robust covariance
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_data.fillna(0)).covariance_ * 252
            
            def portfolio_objective(weights):
                weights = np.abs(weights) / np.sum(np.abs(weights))
                port_return = np.sum(weights * expected_returns)
                port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                # Maximize Sharpe ratio (return/risk)
                sharpe = port_return / np.sqrt(port_var) if port_var > 0 else 0
                return -sharpe
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
            bounds = tuple((0.001, 0.35) for _ in range(len(assets)))
            
            best_result = None
            best_sharpe = -np.inf
            
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(len(assets)))
                
                try:
                    result = minimize(portfolio_objective, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
                    
                    if result.success and -result.fun > best_sharpe:
                        best_result = result
                        best_sharpe = -result.fun
                except:
                    continue
            
            if best_result and best_result.success:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                return dict(zip(assets, weights))
            
            return self.equal_weight_allocation(assets)
            
        except Exception as e:
            print(f"ML mean-variance optimization failed: {e}")
            return self.equal_weight_allocation(assets)
    
    def regime_adaptive_allocation(self, returns_data, assets):
        """Advanced regime-adaptive allocation strategy"""
        try:
            print("🌍 Regime-Adaptive Allocation...")
            
            # Detect current regime
            current_regime = self.detect_market_regime_hmm(returns_data)
            print(f"   📊 Detected regime: {current_regime}")
            
            # Train return predictor if needed
            if not self.return_predictor:
                self.train_return_predictor(returns_data)
            
            # Regime-specific strategies
            if current_regime in ['bull_market']:
                # Growth-oriented in bull markets
                print("   📈 Using growth-oriented strategy")
                return self.ml_enhanced_black_litterman(returns_data, assets, confidence_level=0.7)
                
            elif current_regime in ['bear_market', 'crisis']:
                # Defensive in bear markets
                print("   🛡️ Using defensive strategy")
                return self.advanced_minimum_variance(returns_data, assets)
                
            elif current_regime in ['high_volatility']:
                # Volatility targeting in high vol environments
                print("   ⚡ Using volatility-targeting strategy")
                return self.advanced_volatility_target_allocation(returns_data, assets, target_vol=0.12)
                
            else:
                # Balanced approach for normal/sideways markets
                print("   ⚖️ Using balanced strategy")
                return self.ml_enhanced_black_litterman(returns_data, assets, confidence_level=0.8)
                
        except Exception as e:
            print(f"Regime-adaptive allocation failed: {e}")
            return self.ml_enhanced_black_litterman(returns_data, assets)
    
    def advanced_minimum_variance(self, returns_data, assets):
        """Advanced minimum variance with robust estimation"""
        try:
            # Use multiple covariance estimators and select best
            estimators = {
                'ledoit_wolf': LedoitWolf(),
                'oas': OAS(),
                'empirical': EmpiricalCovariance()
            }
            
            best_cov = None
            best_condition = np.inf
            
            for name, estimator in estimators.items():
                try:
                    cov_matrix = estimator.fit(returns_data.fillna(0)).covariance_
                    condition_num = np.linalg.cond(cov_matrix)
                    if condition_num < best_condition:
                        best_condition = condition_num
                        best_cov = cov_matrix
                except:
                    continue
            
            if best_cov is None:
                best_cov = returns_data.cov().values
            
            def portfolio_variance(weights):
                weights = np.abs(weights) / np.sum(np.abs(weights))
                return np.dot(weights.T, np.dot(best_cov, weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
            bounds = tuple((0.005, 0.35) for _ in range(len(assets)))
            
            best_result = None
            best_variance = np.inf
            
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(len(assets)))
                
                try:
                    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
                    
                    if result.success and result.fun < best_variance:
                        best_result = result
                        best_variance = result.fun
                except:
                    continue
            
            if best_result and best_result.success:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                return dict(zip(assets, weights))
            
            return self.equal_weight_allocation(assets)

        except Exception as e:
            print(f"Advanced minimum variance failed: {e}")
            return self.equal_weight_allocation(assets)
    def advanced_volatility_target_allocation(self, returns_data, assets, target_vol=0.15):
        """Advanced volatility targeting with regime awareness"""
        try:
            # Robust covariance estimation
            lw = LedoitWolf()
            robust_cov = lw.fit(returns_data.fillna(0)).covariance_ * 252
            
            # ML-predicted expected returns
            expected_returns = self.predict_expected_returns(returns_data)
            
            def portfolio_objective(weights):
                weights = np.abs(weights) / np.sum(np.abs(weights))
                port_return = np.sum(weights * expected_returns)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(robust_cov, weights)))
                
                # Penalty for deviating from target volatility
                vol_penalty = 100 * (port_vol - target_vol) ** 2
                
                # Maximize return subject to volatility constraint
                return -(port_return - vol_penalty)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
            bounds = tuple((0.005, 0.40) for _ in range(len(assets)))
            
            best_result = None
            best_objective = -np.inf
            
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(len(assets)))
                
                try:
                    result = minimize(portfolio_objective, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
                    
                    if result.success and -result.fun > best_objective:
                        best_result = result
                        best_objective = -result.fun
                except:
                    continue
            
            if best_result and best_result.success:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                return dict(zip(assets, weights))
            
            return self.equal_weight_allocation(assets)

        except Exception as e:
            print(f"Advanced minimum variance failed: {e}")
            return self.equal_weight_allocation(assets)
    
    # Legacy interface methods
    def equal_weight_allocation(self, assets):
        n_assets = len(assets)
        weights = np.array([1.0 / n_assets] * n_assets)
        return dict(zip(assets, weights))
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }


# Update main class to use advanced version
class EnhancedPortfolioAllocationEngine(AdvancedPortfolioAllocationEngine):
    def __init__(self):
        super().__init__()
    
    # Map legacy methods to new advanced methods
    def enhanced_risk_parity_allocation(self, returns_data, assets):
        return self.regime_adaptive_allocation(returns_data, assets)
    
    def enhanced_maximum_sharpe_allocation(self, returns_data, assets, risk_free_rate=0.02):
        return self.ml_enhanced_black_litterman(returns_data, assets)
    
    def minimum_variance_allocation(self, returns_data, assets):
        return self.advanced_minimum_variance(returns_data, assets)
    
    def enhanced_volatility_target_allocation(self, returns_data, assets, target_vol=0.15):
        return self.advanced_volatility_target_allocation(returns_data, assets, target_vol)
    
    def regime_aware_allocation(self, returns_data, assets, regime=None):
        return self.regime_adaptive_allocation(returns_data, assets)
    
    def enhanced_risk_parity_allocation(self, returns_data, assets):
        """Enhanced risk parity with robust covariance and better optimization"""
        try:
            robust_cov = self.robust_covariance_estimation(returns_data)
            n_assets = len(assets)
            
            def risk_parity_objective(weights):
                weights = np.abs(weights)  # Ensure positive weights
                weights = weights / weights.sum() if weights.sum() > 0 else weights  # Normalize
                
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(robust_cov.values, weights)))
                if portfolio_vol < 1e-8:
                    return 1e6
                    
                marginal_contrib = np.dot(robust_cov.values, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                
                return np.sum((contrib - target_contrib) ** 2)
            
            # Multiple attempts with different starting points
            best_result = None
            best_objective = float('inf')
            
            for seed in [42, 123, 456]:  # Try multiple random starts
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(n_assets))  # Random valid start
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
                bounds = tuple((0.005, 0.40) for _ in range(n_assets))  # Tighter bounds
                
                try:
                    result = minimize(risk_parity_objective, initial_guess, 
                                    method='SLSQP', bounds=bounds, constraints=constraints,
                                    options={'maxiter': 2000, 'ftol': 1e-9})
                    
                    if result.success and result.fun < best_objective:
                        best_result = result
                        best_objective = result.fun
                        
                except Exception as inner_e:
                    continue
            
            if best_result and best_result.success:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                # Ensure minimum diversification
                if np.max(weights) < 0.80:  # No single asset > 80%
                    return dict(zip(assets, weights))
            
        except Exception as e:
            print(f"Enhanced risk parity failed: {e}")
        
        # Fallback: simple inverse volatility
        try:
            vols = returns_data.std()
            inv_vol_weights = (1 / vols) / (1 / vols).sum()
            return dict(zip(assets, inv_vol_weights))
        except:
            return self.equal_weight_allocation(assets)
    
    def regime_aware_allocation(self, returns_data, assets, regime=None):
        """Allocation strategy that adapts to market regime"""
        if regime is None:
            regime = self.detect_market_regime(returns_data)
        
        print(f"📈 Detected regime: {regime}")
        
        if regime == 'crisis':
            # In crisis: minimize risk, equal weights with bonds proxy
            return self.minimum_variance_allocation(returns_data, assets)
        elif regime == 'bear_market':
            # In bear market: defensive allocation
            return self.enhanced_risk_parity_allocation(returns_data, assets)
        elif regime == 'bull_market':
            # In bull market: growth-oriented
            return self.enhanced_maximum_sharpe_allocation(returns_data, assets)
        elif regime == 'high_volatility':
            # High vol: volatility targeting
            return self.enhanced_volatility_target_allocation(returns_data, assets, target_vol=0.12)
        else:
            # Default: balanced approach
            return self.enhanced_risk_parity_allocation(returns_data, assets)
    
    def enhanced_maximum_sharpe_allocation(self, returns_data, assets, risk_free_rate=0.02):
        """Enhanced maximum Sharpe ratio with robust estimation and better optimization"""
        try:
            # Use robust covariance and mean estimation
            robust_cov = self.robust_covariance_estimation(returns_data)
            
            # Robust mean estimation (trim extreme values)
            returns_trimmed = returns_data.apply(lambda x: stats.trim_mean(x.dropna(), 0.1))
            annual_returns = returns_trimmed * 252
            annual_cov = robust_cov * 252
            
            n_assets = len(assets)
            
            def negative_sharpe(weights):
                weights = np.abs(weights)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                
                portfolio_return = np.sum(weights * annual_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
                
                if portfolio_vol < 1e-8:
                    return 1e6
                    
                sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                return -sharpe
            
            # Multiple optimization attempts
            best_result = None
            best_sharpe = -float('inf')
            
            for seed in [42, 123, 456, 789]:
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(n_assets))
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
                bounds = tuple((0.005, 0.35) for _ in range(n_assets))  # Concentration limits
                
                try:
                    result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints,
                                    options={'maxiter': 2000, 'ftol': 1e-9})
                    
                    if result.success and -result.fun > best_sharpe:
                        best_result = result
                        best_sharpe = -result.fun
                        
                except Exception as inner_e:
                    continue
            
            if best_result and best_result.success and best_sharpe > 0:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                return dict(zip(assets, weights))
                
        except Exception as e:
            print(f"Enhanced max Sharpe failed: {e}")
        
        # Fallback: simple momentum-based allocation
        try:
            returns_3m = returns_data.tail(60).mean() * 252  # 3-month momentum
            positive_returns = returns_3m[returns_3m > 0]
            if len(positive_returns) > 0:
                weights = positive_returns / positive_returns.sum()
                result = {asset: 0 for asset in assets}
                for asset in positive_returns.index:
                    if asset in result:
                        result[asset] = weights[asset]
                return result
        except:
            pass
            
        return self.equal_weight_allocation(assets)
    
    def enhanced_volatility_target_allocation(self, returns_data, assets, target_vol=0.15):
        """Enhanced volatility targeting with leverage control"""
        robust_cov = self.robust_covariance_estimation(returns_data)
        volatilities = np.sqrt(np.diag(robust_cov * 252))
        
        # Inverse volatility weights with correlation adjustment
        inv_vol_weights = 1 / volatilities
        
        # Adjust for correlations
        corr_matrix = robust_cov.corr()
        avg_correlations = corr_matrix.mean()
        
        # Penalize highly correlated assets
        adjusted_weights = inv_vol_weights * (1 - avg_correlations * 0.5)
        adjusted_weights = np.maximum(adjusted_weights, 0.01)  # Minimum weight
        
        # Normalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(np.dot(adjusted_weights.T, np.dot(robust_cov * 252, adjusted_weights)))
        
        # Scale to target volatility
        if portfolio_vol > 0:
            leverage = min(2.0, target_vol / portfolio_vol)  # Max 2x leverage
            final_weights = adjusted_weights * leverage
            
            # Add cash if leverage < 1
            cash_weight = max(0, 1 - final_weights.sum())
            
            result = dict(zip(assets, final_weights))
            if cash_weight > 0:
                result['CASH'] = cash_weight
                
            return result
        
        return self.equal_weight_allocation(assets)
    
    def hierarchical_risk_parity(self, returns_data, assets):
        """Hierarchical Risk Parity allocation"""
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Convert to distance matrix
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Hierarchical clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Build hierarchical allocation (simplified version)
            weights = self._recursive_bisection(corr_matrix.values, returns_data.cov().values)
            
            return dict(zip(assets, weights))
            
        except Exception as e:
            print(f"HRP allocation failed: {e}")
            return self.enhanced_risk_parity_allocation(returns_data, assets)
    
    def _recursive_bisection(self, corr_matrix, cov_matrix):
        """Recursive bisection for HRP"""
        n_assets = len(corr_matrix)
        weights = np.ones(n_assets) / n_assets
        
        # Simplified implementation - just return equal weights
        # Full HRP implementation would require more complex tree traversal
        return weights
    
    # Keep all original methods for compatibility
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def equal_weight_allocation(self, assets):
        n_assets = len(assets)
        weights = np.array([1.0 / n_assets] * n_assets)
        return dict(zip(assets, weights))
    
    def risk_parity_allocation(self, returns_data, assets):
        return self.enhanced_risk_parity_allocation(returns_data, assets)
    
    def maximum_sharpe_allocation(self, returns_data, assets, risk_free_rate=0.02):
        return self.enhanced_maximum_sharpe_allocation(returns_data, assets, risk_free_rate)
    
    def minimum_variance_allocation(self, returns_data, assets):
        """Minimum variance allocation with improved optimization"""
        try:
            robust_cov = self.robust_covariance_estimation(returns_data)
            n_assets = len(assets)
            
            def portfolio_variance(weights):
                weights = np.abs(weights)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                return np.dot(weights.T, np.dot(robust_cov.values, weights))
            
            # Multiple optimization attempts
            best_result = None
            best_variance = float('inf')
            
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                initial_guess = np.random.dirichlet(np.ones(n_assets))
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
                bounds = tuple((0.005, 0.50) for _ in range(n_assets))
                
                try:
                    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints,
                                    options={'maxiter': 2000})
                    
                    if result.success and result.fun < best_variance:
                        best_result = result
                        best_variance = result.fun
                        
                except Exception as inner_e:
                    continue
            
            if best_result and best_result.success:
                weights = np.abs(best_result.x)
                weights = weights / weights.sum()
                return dict(zip(assets, weights))
                
        except Exception as e:
            print(f"Minimum variance optimization failed: {e}")
        
        # Fallback: inverse volatility
        try:
            vols = returns_data.std()
            inv_vol_weights = (1 / vols) / (1 / vols).sum()
            return dict(zip(assets, inv_vol_weights))
        except:
            return self.equal_weight_allocation(assets)
    
    def momentum_based_allocation(self, returns_data, assets, lookback_periods=[20, 60, 120]):
        """Momentum-based allocation with proper risk adjustment"""
        try:
            price_data = (1 + returns_data).cumprod()
            momentum_scores = {}
            
            for asset in assets:
                if asset in price_data.columns:
                    scores = []
                    for period in lookback_periods:
                        if len(price_data) >= period:
                            # Calculate momentum
                            momentum = (price_data[asset].iloc[-1] / price_data[asset].iloc[-period] - 1)
                            # Adjust momentum by volatility (risk-adjusted momentum)
                            vol_adj = returns_data[asset].tail(period).std()
                            if vol_adj > 0:
                                risk_adj_momentum = momentum / (vol_adj * np.sqrt(period))
                            else:
                                risk_adj_momentum = momentum
                            scores.append(risk_adj_momentum)
                    momentum_scores[asset] = np.mean(scores) if scores else 0
                else:
                    momentum_scores[asset] = 0
            
            # Only invest in positive momentum assets
            positive_momentum = {k: v for k, v in momentum_scores.items() if v > 0}
            
            if positive_momentum:
                # Weight by momentum strength
                total_momentum = sum(positive_momentum.values())
                weights = {asset: 0 for asset in assets}
                
                if total_momentum > 0:
                    for asset, score in positive_momentum.items():
                        weights[asset] = score / total_momentum
                    return weights
            
            # Fallback if no positive momentum
            return self.equal_weight_allocation(assets)
            
        except Exception as e:
            print(f"Momentum allocation failed: {e}")
            return self.equal_weight_allocation(assets)
    
    def volatility_target_allocation(self, returns_data, assets, target_volatility=0.15):
        return self.enhanced_volatility_target_allocation(returns_data, assets, target_volatility)
    
    def black_litterman_allocation(self, returns_data, assets, market_caps=None, tau=0.025):
        try:
            if market_caps is None:
                market_caps = {asset: 1.0 for asset in assets}
            
            total_market_cap = sum(market_caps.values())
            prior_weights = np.array([market_caps.get(asset, 0) / total_market_cap for asset in assets])
            
            # Use robust covariance
            robust_cov = self.robust_covariance_estimation(returns_data)
            returns = returns_data.mean() * 252
            cov_matrix = robust_cov * 252
            
            risk_aversion = 3.0
            implied_returns = risk_aversion * np.dot(cov_matrix, prior_weights)
            
            bl_returns = implied_returns
            bl_cov = cov_matrix
            
            def portfolio_utility(weights):
                port_return = np.sum(weights * bl_returns)
                port_var = np.dot(weights.T, np.dot(bl_cov, weights))
                return -(port_return - 0.5 * risk_aversion * port_var)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 0.40) for _ in range(len(assets)))
            
            result = minimize(portfolio_utility, prior_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(assets, result.x))
        except:
            pass
        
        return self.equal_weight_allocation(assets)

# Update the legacy class to use enhanced version
class PortfolioAllocationEngine(EnhancedPortfolioAllocationEngine):
    def __init__(self):
        super().__init__()

portfolio_allocator = PortfolioAllocationEngine()
