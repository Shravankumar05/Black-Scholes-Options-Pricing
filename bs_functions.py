import math

def phi(x):
    return 0.5 * (1+ math.erf(x / math.sqrt(2)))

def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    return (S * phi(d1)) - (K * math.exp(-r * T) * phi(d2))


def black_scholes_puts(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    return (K * math.exp(-r * T) * phi(-d2)) - (S * phi(-d1))
    

def pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def vega(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    vega_value = S * pdf(d1) * math.sqrt(T)
    return vega_value

def implied_volatility(target_price, S, K, T, r, is_call=True, initial_guess=0.2, tol=1e-6, max_iter=100):
    sigma = initial_guess
    for i in range(max_iter):
        price = (black_scholes_call if is_call else black_scholes_puts)(S, K, T, r, sigma)
        diff = price - target_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:
            raise RuntimeError(f"Vega too small ({v}) – cannot converge.")
        sigma -= diff / v
        if sigma <= 0:
            sigma = tol
    raise RuntimeError(f"Implied vol did not converge after {max_iter} iterations")

def delta_call(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    return phi(d1)

def delta_put(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    return phi(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    return pdf(d1) / (S * sigma * math.sqrt(T))

def theta_put(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    theta = (-S * pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * phi(-d2))
    return theta

def theta_call(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    theta = (-S * pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * phi(d2))
    return theta

