def test_ccxt_installation():
    try:
        import ccxt
        print(f"CCXT imported successfully")
        print(f"CCXT version: {ccxt.__version__}")
        
        exchange = ccxt.binance()
        print(f"Binance exchange created")
        
        try:
            markets = exchange.load_markets()
            print(f"Markets loaded: {len(markets)} available")
            
            popular_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol][:10]
            print(f"Sample USDT pairs: {popular_pairs}")
            
            try:
                ticker = exchange.fetch_ticker('BTC/USDT')
                print(f"BTC/USDT price: ${ticker['last']:,.2f}")
                return True
                
            except Exception as e:
                print(f"Price fetch failed: {e}")
                return False
                
        except Exception as e:
            print(f"Market loading failed: {e}")
            print("This might be due to network issues or API limits")
            return False
            
    except ImportError as e:
        print(f"CCXT import failed: {e}")
        print("Install with: pip install ccxt")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_ccxt_installation()
    if success:
        print("\n All tests passed! Crypto functionality should work.")
    else:
        print("\n❌Some tests failed L. Check the errors above.")