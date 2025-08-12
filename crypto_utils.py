import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import ccxt
    print(f"crypto_utils.py: CCXT imported successfully, version: {ccxt.__version__}")
except ImportError as e:
    print(f"crypto_utils.py: Failed to import CCXT: {e}")
    raise

class CryptoDataFetcher:
    def __init__(self):
        try:
            self.exchange = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': False,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'timeout': 30000,
                'headers': {
                    'User-Agent': 'ccxt/python'
                }
            })
            self.exchange.load_markets()
            print("✅ Binance connection established")
        except Exception as e:
            print(f"⚠️ Binance connection failed: {e}")
            print("Using fallback prices for crypto")
            self.exchange = None
    
    def fetch_crypto_ohlcv(self, symbol, timeframe='1d', days=365):
        if not self.exchange:
            return self._generate_synthetic_crypto_data(symbol, days)
        
        try:
            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=min(1000, days))
            
            if not ohlcv or len(ohlcv) < 10:
                print(f"⚠️ Insufficient data for {symbol}, using synthetic data")
                return self._generate_synthetic_crypto_data(symbol, days)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            print(f"✅ Fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            print(f"⚠️ API error for {symbol}: {e}, using synthetic data")
            return self._generate_synthetic_crypto_data(symbol, days)
    
    def _generate_synthetic_crypto_data(self, symbol, days): # VaR
        try:
            base_prices = {
                'BTC/USDT': 43500.0, 'ETH/USDT': 2650.0, 'BNB/USDT': 310.0,
                'ADA/USDT': 0.52, 'SOL/USDT': 105.0, 'XRP/USDT': 0.63,
                'DOT/USDT': 7.2, 'DOGE/USDT': 0.082, 'AVAX/USDT': 37.0,
                'MATIC/USDT': 0.92
            }
            
            base_price = base_prices.get(symbol, 100.0)
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
            np.random.seed(42)
            daily_returns = np.random.normal(0.001, 0.04, days)  # 4% daily volatility
            prices = [base_price]
            for ret in daily_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(df)))
            df['volume'] = np.random.uniform(1000000, 10000000, len(df))
            
            print(f"📊 Generated synthetic data for {symbol}: {len(df)} days")
            return df
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        fallback_prices = {
            'BTC/USDT': 100000.0, 'ETH/USDT': 4400.0, 'BNB/USDT': 310.0,
            'ADA/USDT': 0.52, 'SOL/USDT': 105.0, 'XRP/USDT': 0.63,
            'DOT/USDT': 7.2, 'DOGE/USDT': 0.082, 'AVAX/USDT': 37.0,
            'MATIC/USDT': 0.92, 'LINK/USDT': 15.5, 'UNI/USDT': 6.8,
            'LTC/USDT': 73.0, 'BCH/USDT': 245.0, 'ATOM/USDT': 8.1
        }
        
        if not self.exchange:
            return fallback_prices.get(symbol, 100.0)
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
            print(f"✅ Fetched {symbol}: ${price:,.2f}")
            return price
        except Exception as e:
            print(f"⚠️ API failed for {symbol}, using fallback: {e}")
            return fallback_prices.get(symbol, 100.0)
    
    def calculate_crypto_volatility(self, symbol, days=365):
        try:
            df = self.fetch_crypto_ohlcv(symbol, '1d', days)
            
            if df.empty or len(df) < 10:
                return 0.5
            
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            if len(returns) == 0:
                return 0.5
            
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(365)
            return float(annual_vol)
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.5
    
    def get_supported_symbols(self):
        popular_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT',
            'FTT/USDT', 'NEAR/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT'
        ]
        
        if not self.exchange:
            return popular_symbols
        
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol and markets[symbol]['active']]
            return sorted(usdt_pairs[:50])  # Return top 50 active USDT pairs
        except:
            return popular_symbols
    
    def is_crypto_symbol(self, symbol):
        crypto_indicators = ['/USDT', '/BUSD', '/BTC', '/ETH', '/BNB']
        return any(indicator in symbol.upper() for indicator in crypto_indicators)

crypto_fetcher = CryptoDataFetcher()
def fetch_crypto_data(symbols, period="1y"):
    if isinstance(symbols, str):
        symbols = [symbols]
    
    period_days = {'1y': 365, '6m': 180, '3m': 90, '1m': 30, '1w': 7, '5d': 5}.get(period, 365)
    all_data = {}
    
    for symbol in symbols:
        if crypto_fetcher.is_crypto_symbol(symbol):
            df = crypto_fetcher.fetch_crypto_ohlcv(symbol, '1d', period_days)
            if not df.empty:
                all_data[symbol] = df['close']
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.DataFrame(all_data)
    return combined_df.dropna()

def get_crypto_current_price(symbol):
    return crypto_fetcher.get_current_price(symbol)

def calculate_crypto_returns(price_data):
    if price_data.empty:
        return pd.DataFrame()
    
    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna()