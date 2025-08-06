import time
import numpy as np
from bs_functions import black_scholes_call, black_scholes_puts, black_scholes_vectorized, black_scholes_multithreaded, black_scholes_optimized, black_scholes_jit, phi, pdf, NUMBA_AVAILABLE

def test_performance():
    K = 100
    T = 1.0
    r = 0.05
    N = 400
    
    S_min, S_max = 50, 150
    sigma_min, sigma_max = 0.1, 0.5
    S_grid = np.linspace(S_min, S_max, num=N)
    sigma_grid = np.linspace(sigma_min, sigma_max, num=N)
    S_mat, sigma_mat = np.meshgrid(S_grid, sigma_grid)
    
    print(f"Testing with {N}x{N} = {N*N} calculations")
    print("-" * 50)
    
    # Original methods
    start_time = time.time()
    call_prices_loop = np.zeros((N, N))
    put_prices_loop = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            call_prices_loop[i, j] = black_scholes_call(S_mat[i, j], K, T, r, sigma_mat[i, j])
            put_prices_loop[i, j] = black_scholes_puts(S_mat[i, j], K, T, r, sigma_mat[i, j])
    
    loop_time = time.time() - start_time
    print(f"Original loop method: {loop_time:.4f} seconds")
    
    # Vectorized method pre-improv
    start_time = time.time()
    d1 = (np.log(S_mat / K) + (r + 0.5 * sigma_mat**2) * T) / (sigma_mat * np.sqrt(T))
    d2 = d1 - sigma_mat * np.sqrt(T)
    
    vec_phi = np.vectorize(phi)
    vec_pdf = np.vectorize(pdf)
    Phi_d1 = vec_phi(d1)
    Phi_d2 = vec_phi(d2)
    Phi_m_d1 = vec_phi(-d1)
    Phi_m_d2 = vec_phi(-d2)
    call_prices_old_vec = S_mat * Phi_d1 - K * np.exp(-r * T) * Phi_d2
    put_prices_old_vec = K * np.exp(-r * T) * Phi_m_d2 - S_mat * Phi_m_d1
    old_vec_time = time.time() - start_time
    print(f"Old vectorized method: {old_vec_time:.4f} seconds")
    
    # Optimized vectorized method
    start_time = time.time()
    call_prices_new_vec = black_scholes_vectorized(S_mat, K, T, r, sigma_mat, 'call')
    put_prices_new_vec = black_scholes_vectorized(S_mat, K, T, r, sigma_mat, 'put')
    new_vec_time = time.time() - start_time
    print(f"New vectorized method: {new_vec_time:.4f} seconds")
    
    # Multi-threaded method (2 cores)
    start_time = time.time()
    call_prices_mt = black_scholes_multithreaded(S_mat, K, T, r, sigma_mat, 'call', n_threads=2)
    put_prices_mt = black_scholes_multithreaded(S_mat, K, T, r, sigma_mat, 'put', n_threads=2)
    mt_time = time.time() - start_time
    print(f"Multi-threaded method (2 cores): {mt_time:.4f} seconds")
    
    # JIT-compiled method
    if NUMBA_AVAILABLE:
        _ = black_scholes_jit(S_mat[:10, :10], K, T, r, sigma_mat[:10, :10], 'call')
        start_time = time.time()
        call_prices_jit = black_scholes_jit(S_mat, K, T, r, sigma_mat, 'call')
        put_prices_jit = black_scholes_jit(S_mat, K, T, r, sigma_mat, 'put')
        jit_time = time.time() - start_time
        print(f"JIT-compiled method: {jit_time:.4f} seconds")
    else:
        print("JIT-compiled method: Not available (install numba)")
        jit_time = float('inf')
        call_prices_jit = call_prices_new_vec
        put_prices_jit = put_prices_new_vec
    
    # Adaptive
    start_time = time.time()
    call_prices_opt = black_scholes_optimized(S_mat, K, T, r, sigma_mat, 'call')
    put_prices_opt = black_scholes_optimized(S_mat, K, T, r, sigma_mat, 'put')
    opt_time = time.time() - start_time
    print(f"Optimized method (adaptive): {opt_time:.4f} seconds")
    
    print("-" * 50)
    best_time = min(new_vec_time, mt_time, jit_time, opt_time)
    print(f"Speedup vs loops: {loop_time / best_time:.1f}x")
    print(f"Speedup vs old vectorized: {old_vec_time / best_time:.1f}x")
    if NUMBA_AVAILABLE:
        print(f"JIT vs vectorized: {new_vec_time / jit_time:.1f}x faster")
        print(f"JIT vs multi-threaded: {mt_time / jit_time:.1f}x faster")
    
    # Verify results are the same
    call_diff_vec = np.max(np.abs(call_prices_loop - call_prices_new_vec))
    put_diff_vec = np.max(np.abs(put_prices_loop - put_prices_new_vec))
    call_diff_mt = np.max(np.abs(call_prices_loop - call_prices_mt))
    put_diff_mt = np.max(np.abs(put_prices_loop - put_prices_mt))
    call_diff_opt = np.max(np.abs(call_prices_loop - call_prices_opt))
    put_diff_opt = np.max(np.abs(put_prices_loop - put_prices_opt))
    
    if NUMBA_AVAILABLE:
        call_diff_jit = np.max(np.abs(call_prices_loop - call_prices_jit))
        put_diff_jit = np.max(np.abs(put_prices_loop - put_prices_jit))
    else:
        call_diff_jit = 0.0
        put_diff_jit = 0.0
    
    print(f"\nMax difference in call prices (vectorized): {call_diff_vec:.2e}")
    print(f"Max difference in put prices (vectorized): {put_diff_vec:.2e}")
    print(f"Max difference in call prices (multi-threaded): {call_diff_mt:.2e}")
    print(f"Max difference in put prices (multi-threaded): {put_diff_mt:.2e}")
    if NUMBA_AVAILABLE:
        print(f"Max difference in call prices (JIT): {call_diff_jit:.2e}")
        print(f"Max difference in put prices (JIT): {put_diff_jit:.2e}")
    print(f"Max difference in call prices (optimized): {call_diff_opt:.2e}")
    print(f"Max difference in put prices (optimized): {put_diff_opt:.2e}")
    
    all_diffs_ok = (call_diff_vec < 1e-10 and put_diff_vec < 1e-10 and 
                    call_diff_mt < 1e-10 and put_diff_mt < 1e-10 and
                    call_diff_opt < 1e-10 and put_diff_opt < 1e-10 and
                    call_diff_jit < 1e-10 and put_diff_jit < 1e-10)
    
    if all_diffs_ok:
        print("Results are good")
    else:
        print("These results mismatch what's expected")

if __name__ == "__main__":
    test_performance()