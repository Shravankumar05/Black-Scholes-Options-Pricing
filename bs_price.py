from bs_pricer import black_scholes_call, black_scholes_puts

if __name__ == "__main__":
    print("Enter current stock price S:")
    S = float(input())

    print("Enter strike price K:")
    K = float(input())

    print("Enter time to expiry in years T:")
    T = float(input())

    print("Enter volatility sigma:")
    sigma = float(input())

    print("Enter risk-free interest rate r:")
    r = float(input())

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_puts(S, K, T, r, sigma)

    print(f"\nCall price: {call_price:.4f}")
    print(f"Put price:  {put_price:.4f}")