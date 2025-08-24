import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from bs_functions import implied_volatility

class OptionsMarketData:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300
    
    def fetch_options_chain(self, symbol, days_to_expiry=None):
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return None
            
            if days_to_expiry:
                target_date = datetime.now() + timedelta(days=days_to_expiry)
                closest_exp = min(expirations, 
                                key=lambda x: abs((pd.to_datetime(x) - target_date).days))
            else:
                closest_exp = expirations[0]
            
            options_data = ticker.option_chain(closest_exp)
            calls = options_data.calls
            puts = options_data.puts
            
            stock_info = ticker.history(period="1d")
            current_price = stock_info['Close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'expiration': closest_exp,
                'calls': calls,
                'puts': puts,
                'days_to_expiry': (pd.to_datetime(closest_exp) - datetime.now()).days
            }
            
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return None
    
    def calculate_implied_volatility_surface(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options[:4]
            
            iv_surface = []
            
            for exp in expirations:
                options_data = ticker.option_chain(exp)
                calls = options_data.calls
                
                stock_info = ticker.history(period="1d")
                S = stock_info['Close'].iloc[-1]
                
                days_to_exp = (pd.to_datetime(exp) - datetime.now()).days
                T = days_to_exp / 365.0
                r = 0.05
                
                for _, option in calls.iterrows():
                    try:
                        K = option['strike']
                        market_price = (option['bid'] + option['ask']) / 2
                        
                        if market_price > 0.01:
                            iv = implied_volatility(market_price, S, K, T, r, is_call=True)
                            
                            iv_surface.append({
                                'expiration': exp,
                                'strike': K,
                                'days_to_expiry': days_to_exp,
                                'time_to_expiry': T,
                                'moneyness': K/S,
                                'implied_vol': iv,
                                'market_price': market_price,
                                'bid': option['bid'],
                                'ask': option['ask'],
                                'volume': option['volume'],
                                'open_interest': option['openInterest']
                            })
                    except:
                        continue
            
            return pd.DataFrame(iv_surface)
            
        except Exception as e:
            print(f"Error calculating IV surface for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_options_flow(self, symbol):
        try:
            chain_data = self.fetch_options_chain(symbol)
            if not chain_data:
                return None
            
            calls = chain_data['calls']
            puts = chain_data['puts']
            
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            
            calls['vol_oi_ratio'] = calls['volume'] / (calls['openInterest'] + 1)
            puts['vol_oi_ratio'] = puts['volume'] / (puts['openInterest'] + 1)
            
            unusual_calls = calls[calls['vol_oi_ratio'] > 0.5].sort_values('volume', ascending=False)
            unusual_puts = puts[puts['vol_oi_ratio'] > 0.5].sort_values('volume', ascending=False)
            
            return {
                'symbol': symbol,
                'put_call_ratio': put_call_ratio,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'unusual_calls': unusual_calls.head(5),
                'unusual_puts': unusual_puts.head(5),
                'expiration': chain_data['expiration'],
                'current_price': chain_data['current_price']
            }
            
        except Exception as e:
            print(f"Error analyzing options flow for {symbol}: {e}")
            return None

options_market_data = OptionsMarketData()