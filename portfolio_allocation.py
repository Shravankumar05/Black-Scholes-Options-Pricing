import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioAllocationEngine:
    def __init__(self):
        self.risk_tolerance_profiles = {
            'conservative': {'max_volatility': 0.12, 'target_return': 0.06},
            'moderate': {'max_volatility': 0.18, 'target_return': 0.09},
            'aggressive': {'max_volatility': 0.30, 'target_return': 0.15}
        }
    
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
        cov_matrix = returns_data.cov().values
        n_assets = len(assets)
        
        def risk_parity_objective(weights, cov_matrix):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - contrib.mean()) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(risk_parity_objective, initial_guess, 
                            args=(cov_matrix,), method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.success:
                return dict(zip(assets, result.x))
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def maximum_sharpe_allocation(self, returns_data, assets, risk_free_rate=0.02):
        returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        n_assets = len(assets)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.success:
                return dict(zip(assets, result.x))
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def minimum_variance_allocation(self, returns_data, assets):
        cov_matrix = returns_data.cov().values
        n_assets = len(assets)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.success:
                return dict(zip(assets, result.x))
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def momentum_based_allocation(self, returns_data, assets, lookback_periods=[20, 60, 120]):
        try:
            price_data = (1 + returns_data).cumprod()
            momentum_scores = {}
            
            for asset in assets:
                if asset in price_data.columns:
                    scores = []
                    for period in lookback_periods:
                        if len(price_data) >= period:
                            momentum = (price_data[asset].iloc[-1] / price_data[asset].iloc[-period] - 1)
                            scores.append(momentum)
                    momentum_scores[asset] = np.mean(scores) if scores else 0
                else:
                    momentum_scores[asset] = 0
            
            # Normalize scores to weights
            total_positive_momentum = sum(max(0, score) for score in momentum_scores.values())
            if total_positive_momentum > 0:
                weights = {asset: max(0, score) / total_positive_momentum 
                          for asset, score in momentum_scores.items()}
                
                # Ensure weights sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {asset: weight / total_weight for asset, weight in weights.items()}
                    return weights
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def volatility_target_allocation(self, returns_data, assets, target_volatility=0.15):
        try:
            volatilities = returns_data.std() * np.sqrt(252)
            
            # Inverse volatility weighting with target adjustment
            inv_vol_weights = {}
            for asset in assets:
                if asset in volatilities.index:
                    vol = volatilities[asset]
                    inv_vol_weights[asset] = (1 / vol) if vol > 0 else 0
                else:
                    inv_vol_weights[asset] = 0
            
            # Normalize
            total_inv_vol = sum(inv_vol_weights.values())
            if total_inv_vol > 0:
                weights = {asset: weight / total_inv_vol for asset, weight in inv_vol_weights.items()}
                
                # Scale to target volatility
                current_vol = self._calculate_portfolio_volatility(weights, returns_data)
                if current_vol > 0:
                    scale_factor = min(1.0, target_volatility / current_vol)
                    weights = {asset: weight * scale_factor for asset, weight in weights.items()}
                    
                    # Add cash allocation if scaled down
                    cash_weight = 1 - sum(weights.values())
                    if cash_weight > 0:
                        weights['CASH'] = cash_weight
                    
                    return weights
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def _calculate_portfolio_volatility(self, weights, returns_data):
        try:
            asset_weights = np.array([weights.get(asset, 0) for asset in returns_data.columns])
            cov_matrix = returns_data.cov() * 252
            return np.sqrt(np.dot(asset_weights.T, np.dot(cov_matrix, asset_weights)))
        except:
            return 0.15
    
    def black_litterman_allocation(self, returns_data, assets, market_caps=None, tau=0.025):
        try:
            if market_caps is None:
                # Use equal market caps if not provided
                market_caps = {asset: 1.0 for asset in assets}
            
            # Market capitalization weights (prior)
            total_market_cap = sum(market_caps.values())
            prior_weights = np.array([market_caps.get(asset, 0) / total_market_cap for asset in assets])
            
            # Historical returns and covariance
            returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical assumption
            implied_returns = risk_aversion * np.dot(cov_matrix, prior_weights)
            
            # Black-Litterman formula (no views for simplicity)
            # In practice, you would incorporate investor views here
            bl_returns = implied_returns  # Simplified: no views
            bl_cov = cov_matrix
            
            # Optimize portfolio
            def portfolio_utility(weights):
                port_return = np.sum(weights * bl_returns)
                port_var = np.dot(weights.T, np.dot(bl_cov, weights))
                return -(port_return - 0.5 * risk_aversion * port_var)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(assets)))
            
            result = minimize(portfolio_utility, prior_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(assets, result.x))
        except:
            pass
        
        return self.equal_weight_allocation(assets)
    
    def get_allocation_recommendation(self, returns_data, assets, risk_profile='moderate', 
                                    current_allocations=None, market_regime='normal'):
        
        recommendations = {}
        
        # Calculate different allocation strategies
        allocations = {
            'Equal Weight': self.equal_weight_allocation(assets),
            'Risk Parity': self.risk_parity_allocation(returns_data, assets),
            'Maximum Sharpe': self.maximum_sharpe_allocation(returns_data, assets),
            'Minimum Variance': self.minimum_variance_allocation(returns_data, assets),
            'Momentum Based': self.momentum_based_allocation(returns_data, assets),
            'Volatility Target': self.volatility_target_allocation(returns_data, assets),
            'Black-Litterman': self.black_litterman_allocation(returns_data, assets)
        }
        
        # Calculate metrics for each allocation
        for strategy, allocation in allocations.items():
            asset_weights = np.array([allocation.get(asset, 0) for asset in returns_data.columns])
            if np.sum(asset_weights) > 0:
                metrics = self.calculate_portfolio_metrics(asset_weights, returns_data, returns_data.cov())
                recommendations[strategy] = {
                    'allocation': allocation,
                    'metrics': metrics,
                    'suitability_score': self._calculate_suitability_score(metrics, risk_profile, market_regime)
                }
        
        # Rank strategies by suitability
        ranked_strategies = sorted(recommendations.items(), 
                                 key=lambda x: x[1]['suitability_score'], reverse=True)
        
        return {
            'recommendations': recommendations,
            'ranked_strategies': ranked_strategies,
            'top_recommendation': ranked_strategies[0] if ranked_strategies else None
        }
    
    def _calculate_suitability_score(self, metrics, risk_profile, market_regime):
        profile = self.risk_tolerance_profiles.get(risk_profile, self.risk_tolerance_profiles['moderate'])
        
        score = metrics['sharpe_ratio'] * 10
        vol_penalty = max(0, (metrics['volatility'] - profile['max_volatility']) * 20)
        score -= vol_penalty
        return_bonus = min(5, metrics['return'] / profile['target_return'])
        score += return_bonus
        if market_regime == 'high_volatility':
            score += (0.25 - metrics['volatility']) * 10
        elif market_regime == 'bull_market':
            score += metrics['return'] * 2
        elif market_regime == 'bear_market':
            score += (0.15 - metrics['volatility']) * 15
        
        return max(0, score)
    
    def rebalancing_suggestions(self, current_allocations, target_allocations, threshold=0.05):
        suggestions = []
        
        for asset in set(list(current_allocations.keys()) + list(target_allocations.keys())):
            current = current_allocations.get(asset, 0)
            target = target_allocations.get(asset, 0)
            difference = target - current
            
            if abs(difference) > threshold:
                action = "BUY" if difference > 0 else "SELL"
                suggestions.append({
                    'asset': asset,
                    'action': action,
                    'current_weight': current,
                    'target_weight': target,
                    'difference': difference,
                    'priority': abs(difference)
                })
        
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        return suggestions

portfolio_allocator = PortfolioAllocationEngine()