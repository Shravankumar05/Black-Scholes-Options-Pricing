import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from bs_pricer import black_scholes_call, black_scholes_puts
from db_utils import create_table, insert_calculation, insert_output

st.set_page_config(layout="wide")
st.title("Black–Scholes Options Pricer")

with st.sidebar:
    st.header("Model Inputs")
    S     = st.number_input("Spot price (S)",             min_value=0.0, value=100.0, step=1.0)
    K     = st.number_input("Strike price (K)",           min_value=0.0, value=100.0, step=1.0)
    T     = st.number_input("Time to expiry (T, years)",  min_value=0.0, value=1.0,   step=0.1)
    sigma = st.number_input("Volatility (σ)",             min_value=0.0, value=0.2,   step=0.01)
    r     = st.number_input("Risk-free rate (r)",         min_value=0.0, value=0.05,  step=0.01)

    st.markdown("---")
    st.header("Heatmap Ranges")
    S_min     = st.number_input("Min spot (Sₘᵢₙ)",    min_value=0.0, value=50.0,  step=1.0)
    S_max     = st.number_input("Max spot (Sₘₐₓ)",    min_value=0.0, value=150.0, step=1.0)
    sigma_min = st.number_input("Min vol (σₘᵢₙ)",     min_value=0.01,value=0.1,   step=0.01)
    sigma_max = st.number_input("Max vol (σₘₐₓ)",     min_value=0.01,value=0.5,   step=0.01)

    st.markdown("---")
    st.header("Purchase Prices")
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price  = black_scholes_puts(S, K, T, r, sigma)
    call_buy_price = st.number_input("Call purchase price", min_value=0.0, value=call_price, step=0.01)
    put_buy_price  = st.number_input("Put purchase price",  min_value=0.0, value=put_price,  step=0.01)

    st.markdown("---")
    save = st.button("📊 Generate & Save")

N = 50
S_grid     = np.linspace(S_min,    S_max,    num=N)
sigma_grid = np.linspace(sigma_min, sigma_max, num=N)

call_matrix = np.zeros((N, N))
put_matrix  = np.zeros((N, N))
for i, vol in enumerate(sigma_grid):
    for j, spot in enumerate(S_grid):
        call_matrix[i, j] = black_scholes_call(spot, K, T, r, vol)
        put_matrix[i, j]  = black_scholes_puts(spot, K, T, r, vol)

# P&L matrices
call_pnl_matrix = call_matrix - call_buy_price
put_pnl_matrix  = put_matrix  - put_buy_price
max_abs = max(abs(call_pnl_matrix.min()), abs(call_pnl_matrix.max()),abs(put_pnl_matrix.min()),  abs(put_pnl_matrix.max()))

if save:
    create_table()
    params = {
        "stock_price":    S,
        "strike_price":   K,
        "interest_rate":  r,
        "volatility":     sigma,
        "time_to_expiry": T,
        "call_buy_price": call_buy_price,
        "put_buy_price":  put_buy_price
    }
    calc_id = insert_calculation(params)

    for i, vol in enumerate(sigma_grid):
        for j, spot in enumerate(S_grid):
            insert_output(calc_id, float(vol), float(spot),
                          float(call_pnl_matrix[i, j]), True)
            insert_output(calc_id, float(vol), float(spot),
                          float(put_pnl_matrix[i, j]),  False)
    st.success(f"Saved calc_id={calc_id} with {N*N*2} rows.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option Price")
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(call_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn_r")
    fig1.colorbar(im1, ax=ax1, label="Price")
    ax1.set_xlabel("S"); ax1.set_ylabel("σ")
    st.pyplot(fig1)

with col2:
    st.subheader("Put Option Price")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(put_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn_r")
    fig2.colorbar(im2, ax=ax2, label="Price")
    ax2.set_xlabel("S"); ax2.set_ylabel("σ")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Call P&L")
    fig3, ax3 = plt.subplots()
    im3 = ax3.imshow(call_pnl_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn",
                     vmin=-max_abs, vmax=max_abs)
    fig3.colorbar(im3, ax=ax3, label="P&L")
    ax3.set_xlabel("S"); ax3.set_ylabel("σ")
    st.pyplot(fig3)

with col4:
    st.subheader("Put P&L")
    fig4, ax4 = plt.subplots()
    im4 = ax4.imshow(put_pnl_matrix, origin="lower",
                     extent=[S_min, S_max, sigma_min, sigma_max],
                     aspect="auto", cmap="RdYlGn",
                     vmin=-max_abs, vmax=max_abs)
    fig4.colorbar(im4, ax=ax4, label="P&L")
    ax4.set_xlabel("S"); ax4.set_ylabel("σ")
    st.pyplot(fig4)

st.subheader("Breakeven Call Curve")
i_mid     = N // 2
sigma_mid = sigma_grid[i_mid]
S_vals    = S_grid
pnl_slice = call_pnl_matrix[i_mid, :]

fig5, ax5 = plt.subplots()
ax5.plot(S_vals, pnl_slice)
ax5.axhline(0, color="k", linestyle="--")
ax5.set_xlabel("Spot price S")
ax5.set_ylabel("Call P&L")
ax5.set_title(f"σ = {sigma_mid:.3f}")
st.pyplot(fig5)

