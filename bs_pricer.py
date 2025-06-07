import math
from normal_cdf import phi

def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    call_price = (S * phi(d1)) - (K * math.exp(-r * T) * phi(d2))

    return call_price

def black_scholes_puts(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    d2 = d1 - (sigma * math.sqrt(T))
    put_price = (K * math.exp(-r * T) * phi(-d2)) - (S * phi(-d1))
    
    return put_price
