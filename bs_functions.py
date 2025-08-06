import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from numba import jit, vectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def vectorize(*args, **kwargs):
        def decorator(func):
            return np.vectorize(func)
        return decorator

def phi(x):
    return 0.5 * (1+ math.erf(x / math.sqrt(2)))

def phi_vectorized(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))

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

def pdf_vectorized(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

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

def black_scholes_vectorized(S_array, K, T, r, sigma_array, option_type='call'):
    d1 = (np.log(S_array / K) + (r + 0.5 * sigma_array**2) * T) / (sigma_array * np.sqrt(T))
    d2 = d1 - sigma_array * np.sqrt(T)
    
    if option_type == 'call':
        return S_array * phi_vectorized(d1) - K * np.exp(-r * T) * phi_vectorized(d2)
    else:
        return K * np.exp(-r * T) * phi_vectorized(-d2) - S_array * phi_vectorized(-d1)

def _black_scholes_chunk(S_chunk, sigma_chunk, K, T, r, option_type):
    return black_scholes_vectorized(S_chunk, K, T, r, sigma_chunk, option_type)

def black_scholes_multithreaded(S_array, K, T, r, sigma_array, option_type='call', n_threads=2):
    # For smaller datasets threading overhead outweighs benefits so we only use it for bigger datasets
    total_elements = S_array.size
    if total_elements < 50000:
        return black_scholes_vectorized(S_array, K, T, r, sigma_array, option_type)
    
    original_shape = S_array.shape
    
    S_flat = S_array.flatten()
    sigma_flat = sigma_array.flatten()
    chunk_size = len(S_flat) // n_threads
    chunks = []
    
    for i in range(n_threads):
        start_idx = i * chunk_size
        if i == n_threads - 1:
            end_idx = len(S_flat)
        else:
            end_idx = (i + 1) * chunk_size
        
        S_chunk = S_flat[start_idx:end_idx]
        sigma_chunk = sigma_flat[start_idx:end_idx]
        chunks.append((S_chunk, sigma_chunk))
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        process_chunk = partial(_black_scholes_chunk, K=K, T=T, r=r, option_type=option_type)
        futures = [executor.submit(process_chunk, S_chunk, sigma_chunk) for S_chunk, sigma_chunk in chunks]
        results = [future.result() for future in futures]
    
    combined_result = np.concatenate(results)
    return combined_result.reshape(original_shape)

@jit(nopython=True, cache=True)
def phi_jit(x):
    """JIT-compiled cumulative normal distribution function"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

@jit(nopython=True, cache=True)
def pdf_jit(x):
    """JIT-compiled probability density function"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

@jit(nopython=True, cache=True)
def black_scholes_jit_single(S, K, T, r, sigma, is_call=True):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_call:
        return S * phi_jit(d1) - K * math.exp(-r * T) * phi_jit(d2)
    else:
        return K * math.exp(-r * T) * phi_jit(-d2) - S * phi_jit(-d1)

@jit(nopython=True, cache=True)
def black_scholes_jit_array(S_flat, K, T, r, sigma_flat, is_call=True):
    n = len(S_flat)
    result = np.empty(n)
    
    for i in range(n):
        result[i] = black_scholes_jit_single(S_flat[i], K, T, r, sigma_flat[i], is_call)
    
    return result

def black_scholes_jit(S_array, K, T, r, sigma_array, option_type='call'):
    if not NUMBA_AVAILABLE:
        return black_scholes_vectorized(S_array, K, T, r, sigma_array, option_type)
    
    original_shape = S_array.shape
    S_flat = S_array.flatten()
    sigma_flat = sigma_array.flatten()
    
    is_call = (option_type == 'call')
    result_flat = black_scholes_jit_array(S_flat, K, T, r, sigma_flat, is_call)
    return result_flat.reshape(original_shape)

def black_scholes_optimized(S_array, K, T, r, sigma_array, option_type='call'):
    # Priority order:
    # 1. JIT-compiled - if Numba available - fastest for all sizes
    # 2. Multi-threaded - for large datasets - good for >50k points
    # 3. Vectorized - fallback- - reliable baseline
    if NUMBA_AVAILABLE:
        return black_scholes_jit(S_array, K, T, r, sigma_array, option_type)
    else:
        return black_scholes_multithreaded(S_array, K, T, r, sigma_array, option_type, n_threads=2)
