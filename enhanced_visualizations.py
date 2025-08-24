import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class EnhancedVisualizations:
    def __init__(self):
        self.color_scheme = {
            'calls': '#00D4AA',
            'puts': '#FF6B6B', 
            'profit': '#00D4AA',
            'loss': '#FF6B6B',
            'neutral': '#FFB800',
            'background': '#1E1E1E',
            'text': '#F5F5F5'
        }
    
    def plot_volatility_surface_3d(self, iv_surface_df):
        if iv_surface_df.empty:
            st.warning("No implied volatility data available")
            return
        
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                iv_surface_df['moneyness'],
                iv_surface_df['time_to_expiry'],
                iv_surface_df['implied_vol'],
                c=iv_surface_df['implied_vol'],
                cmap='viridis',
                s=60
            )
            
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Time to Expiry (Years)')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Implied Volatility Surface')
            
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
            plt.tight_layout()
            
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)  # Ensure proper cleanup
        except Exception as e:
            st.error(f"Error plotting volatility surface: {e}")
    
    def plot_options_flow_dashboard(self, flow_data):
        if not flow_data:
            st.warning("No options flow data available")
            return
        
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${flow_data['current_price']:.2f}")
            with col2:
                st.metric("Put/Call Ratio", f"{flow_data['put_call_ratio']:.2f}")
            with col3:
                st.metric("Call Volume", f"{flow_data['total_call_volume']:,.0f}")
            with col4:
                st.metric("Put Volume", f"{flow_data['total_put_volume']:,.0f}")
            
            # Create matplotlib figure for volume bars
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            if not flow_data['unusual_calls'].empty:
                ax1.bar(
                    flow_data['unusual_calls']['strike'],
                    flow_data['unusual_calls']['volume'],
                    color=self.color_scheme['calls'],
                    alpha=0.7
                )
                ax1.set_title('Call Volume by Strike')
                ax1.set_xlabel('Strike Price')
                ax1.set_ylabel('Volume')
                ax1.grid(True, alpha=0.3)
            
            if not flow_data['unusual_puts'].empty:
                ax2.bar(
                    flow_data['unusual_puts']['strike'],
                    flow_data['unusual_puts']['volume'],
                    color=self.color_scheme['puts'],
                    alpha=0.7
                )
                ax2.set_title('Put Volume by Strike')
                ax2.set_xlabel('Strike Price')
                ax2.set_ylabel('Volume')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting options flow: {e}")
    
    def plot_advanced_greeks_dashboard(self, S_range, greeks_data):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Delta plot
            ax1.plot(S_range, greeks_data['delta_call'], 
                    label='Call Delta', color=self.color_scheme['calls'], linewidth=2)
            ax1.plot(S_range, greeks_data['delta_put'], 
                    label='Put Delta', color=self.color_scheme['puts'], linewidth=2)
            ax1.set_title('Delta')
            ax1.set_xlabel('Stock Price')
            ax1.set_ylabel('Delta')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gamma plot
            ax2.plot(S_range, greeks_data['gamma'], 
                    label='Gamma', color=self.color_scheme['neutral'], linewidth=2)
            ax2.set_title('Gamma')
            ax2.set_xlabel('Stock Price')
            ax2.set_ylabel('Gamma')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Theta plot
            ax3.plot(S_range, greeks_data['theta_call'], 
                    label='Call Theta', color=self.color_scheme['calls'], linewidth=2)
            ax3.plot(S_range, greeks_data['theta_put'], 
                    label='Put Theta', color=self.color_scheme['puts'], linewidth=2)
            ax3.set_title('Theta')
            ax3.set_xlabel('Stock Price')
            ax3.set_ylabel('Theta')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Vega plot
            ax4.plot(S_range, greeks_data['vega'], 
                    label='Vega', color=self.color_scheme['neutral'], linewidth=2)
            ax4.set_title('Vega')
            ax4.set_xlabel('Stock Price')
            ax4.set_ylabel('Vega')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting Greeks dashboard: {e}")
    def plot_portfolio_risk_metrics(self, returns_data, allocations):
        """Plot comprehensive portfolio risk visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Returns distribution
            portfolio_returns = (returns_data * allocations).sum(axis=1)
            ax1.hist(portfolio_returns, bins=50, alpha=0.7, color=self.color_scheme['neutral'])
            ax1.set_title('Portfolio Returns Distribution')
            ax1.set_xlabel('Daily Returns')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # 2. Cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            ax2.plot(cumulative_returns.index, cumulative_returns, 
                    color=self.color_scheme['profit'], linewidth=2)
            ax2.set_title('Cumulative Portfolio Performance')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Return')
            ax2.grid(True, alpha=0.3)
            
            # 3. Rolling volatility
            rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252)
            ax3.plot(rolling_vol.index, rolling_vol, 
                    color=self.color_scheme['loss'], linewidth=2)
            ax3.set_title('30-Day Rolling Volatility')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Annualized Volatility')
            ax3.grid(True, alpha=0.3)
            
            # 4. Drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            ax4.fill_between(drawdown.index, drawdown, 0, 
                           color=self.color_scheme['loss'], alpha=0.7)
            ax4.set_title('Portfolio Drawdown')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Drawdown (%)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting portfolio risk metrics: {e}")
    
    def plot_correlation_heatmap_dynamic(self, correlation_matrix, returns_data):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Correlation heatmap
        im1 = ax1.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(correlation_matrix.columns)))
        ax1.set_yticks(range(len(correlation_matrix.index)))
        ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax1.set_yticklabels(correlation_matrix.index)
        ax1.set_title('Asset Correlation Matrix')
        
        # Add text annotations
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax1.text(j, i, f'{correlation_matrix.values[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Rolling correlation plot
        if len(returns_data.columns) >= 2:
            asset1, asset2 = returns_data.columns[0], returns_data.columns[1]
            rolling_corr = returns_data[asset1].rolling(30).corr(returns_data[asset2])
            
            ax2.plot(rolling_corr.index, rolling_corr, 
                    color=self.color_scheme['neutral'], linewidth=2)
            ax2.set_title(f'30-Day Rolling Correlation: {asset1} vs {asset2}')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Correlation')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

enhanced_viz = EnhancedVisualizations()