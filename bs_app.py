import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf

from bs_functions import black_scholes_call, black_scholes_puts, implied_volatility, delta_call, delta_put, gamma, theta_put, theta_call, vega, phi, pdf
from db_utils import create_table, insert_calculation, insert_output

st.set_page_config(layout="wide")
st.title("Black–Scholes Options Pricer")

with st.sidebar:
    st.header("Model Inputs")
    symbol = st.text_input("Optional: Stock symbol (e.g. AAPL)", value="")
    live_price = None
    live_vol   = None

    if symbol:
        try:
            tkr  = yf.Ticker(symbol)
            data = tkr.history(period="1mo", interval="1d")["Close"]
            live_price = data.iloc[-1]
            log_rets   = np.log(data / data.shift(1)).dropna()
            live_vol   = log_rets.std() * np.sqrt(252)
            st.success(f"{symbol.upper()}: Spot = {live_price:.2f}, Vol ≈ {live_vol:.2%}")
        except Exception as e:
            st.error(f"Could not fetch data for '{symbol}': {e}")

    default_spot = float(live_price) if live_price is not None else 100.0
    S = st.number_input(
    "Spot price (S)",
    min_value=0.0,
    value=default_spot,
    step=1.0,
    key="spot_input"
    )

    default_vol = float(live_vol) if live_vol is not None else 0.2
    sigma = st.number_input(
        "Volatility (σ)",
        min_value=0.0,
        value=default_vol,
        step=0.001,
        format="%.3f",
        key="vol_input"
    )
    K = st.number_input("Strike price (K)",           min_value=0.0, value=100.0, step=1.0)
    T = st.number_input("Time to expiry (T, years)",  min_value=0.0, value=1.0,   step=0.1)
    r = st.number_input("Risk-free rate (r)",         min_value=0.0, value=0.05,  step=0.01)

    st.divider()
    st.header("Heatmap Ranges")
    S_min     = st.number_input("Min spot (Sₘᵢₙ)",    min_value=0.0, value=50.0,  step=1.0)
    S_max     = st.number_input("Max spot (Sₘₐₓ)",    min_value=0.0, value=150.0, step=1.0)
    sigma_min = st.number_input("Min vol (σₘᵢₙ)",     min_value=0.01,value=0.1,   step=0.01)
    sigma_max = st.number_input("Max vol (σₘₐₓ)",     min_value=0.01,value=0.5,   step=0.01)

    st.divider()
    st.header("Purchase Prices")
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price  = black_scholes_puts(S, K, T, r, sigma)
    call_buy_price = st.number_input("Call purchase price", min_value=0.0, value=call_price, step=0.01)
    put_buy_price  = st.number_input("Put purchase price",  min_value=0.0, value=put_price,  step=0.01)

    imp_vol_call = None
    imp_vol_put  = None

    if call_buy_price > 0:
        try:
            imp_vol_call = implied_volatility(
                call_buy_price, S, K, T, r, is_call=True
            )
        except RuntimeError as e:
            imp_vol_call = f"{e}"

    if put_buy_price > 0:
        try:
            imp_vol_put = implied_volatility(
                put_buy_price, S, K, T, r, is_call=False
            )
        except RuntimeError as e:
            imp_vol_put = f"{e}"

    st.divider()
    st.header("Generate and Download Data")
    st.markdown("Click the button to generate the heatmaps and save their data to a SQLite database. You can then download the database file.")
    save = st.button("📊 Generate & Save to DB")

    if os.path.exists("data.db"):
        with open("data.db", "rb") as f:
            db_bytes = f.read()
        st.download_button(
            label="📥 Download SQLite DB",
            data=db_bytes,
            file_name="data.db",
            mime="application/x-sqlite3"
        )
    else:
        st.warning("No database found. Click ‘Generate & Save’ first.")

N = 300
S_grid     = np.linspace(S_min,    S_max,    num=N)
sigma_grid = np.linspace(sigma_min, sigma_max, num=N)

S_mat, sigma_mat = np.meshgrid(S_grid, sigma_grid)
d1 = (np.log(S_mat / K) + (r + 0.5 * sigma_mat**2) * T) / (sigma_mat * np.sqrt(T))
d2 = d1 - sigma_mat * np.sqrt(T)

vec_phi = np.vectorize(phi)
vec_pdf = np.vectorize(pdf)

Phi_d1 = vec_phi(d1)
Phi_d2 = vec_phi(d2)
Phi_m_d1 = vec_phi(-d1)
Phi_m_d2 = vec_phi(-d2)
pdf_d1 = vec_pdf(d1)

call_matrix = S_mat * Phi_d1 - K * np.exp(-r * T) * Phi_d2
put_matrix  = K * np.exp(-r * T) * Phi_m_d2 - S_mat * Phi_m_d1
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

st.header("1️⃣ Options-Price Heatmaps")
st.info("Enter your model parameters in the sidebar to observe how the call and put value varies with spot price and volatility.")
col0, col1 = st.columns(2)
with col0:
    st.metric(label="Call Option Price", value=f"{call_price:.4f}", border= True)
with col1:
    st.metric(label="Put Option Price",  value=f"{put_price:.4f}", border= True)
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

st.header("2️⃣ P&L Heatmaps")
st.info("Enter your purchase prices in the sidebar to observe how the P&L varies with spot price and volatility. Also the implied volatility from the market option values that you have provided has been given.")

col0, col1 = st.columns(2)
col2, col3 = st.columns(2)

with col2:
    label = "Call Implied Volatility"
    value = f"{imp_vol_call:.4f}" if isinstance(imp_vol_call, float) else imp_vol_call or "–"
    st.metric(label, value, border=True)
with col3:
    label = "Put Implied Volatility"
    value = f"{imp_vol_put:.4f}" if isinstance(imp_vol_put, float) else imp_vol_put or "–"
    st.metric(label, value, border=True)

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

st.header("3️⃣ Break-Even Curves")
st.info("These plots show P&L vs. spot price at the midpoint volatility of your heatmaps. The dashed horizontal line is P&L = 0 (your breakeven point).")
i_mid     = N // 2
sigma_mid = sigma_grid[i_mid]
S_vals    = S_grid
call_pnl_slice = call_pnl_matrix[i_mid, :]
put_pnl_slice  = put_pnl_matrix[i_mid, :]
colA, colB = st.columns(2)

with colA:
    st.subheader(f"Call Break-Even Curve (σ = {sigma_mid:.3f})")
    fig_call, ax_call = plt.subplots()
    ax_call.plot(S_vals, call_pnl_slice, label="Call P&L")
    ax_call.axhline(0, color="k", linestyle="--", label="Breakeven")
    ax_call.set_xlabel("Spot price S")
    ax_call.set_ylabel("P&L")
    ax_call.legend()
    st.pyplot(fig_call)

with colB:
    st.subheader(f"Put Break-Even Curve (σ = {sigma_mid:.3f})")
    fig_put, ax_put = plt.subplots()
    ax_put.plot(S_vals, put_pnl_slice, label="Put P&L")
    ax_put.axhline(0, color="k", linestyle="--", label="Breakeven")
    ax_put.set_xlabel("Spot price S")
    ax_put.set_ylabel("P&L")
    ax_put.legend()
    st.pyplot(fig_put)


st.header("4️⃣ Greeks")
st.info("The Greeks measure the sensitivity of option prices to the input factors. The values below are calculated at the current model inputs.")

greek_call = {
    "Delta": delta_call(S, K, T, r, sigma),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega":  vega(S, K, T, r, sigma),
    "Theta": theta_call(S, K, T, r, sigma)
}
greek_put = {
    "Delta": delta_put(S, K, T, r, sigma),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega":  vega(S, K, T, r, sigma),
    "Theta": theta_put(S, K, T, r, sigma)
}

cols_call = st.columns(4)
for col, (name, val) in zip(cols_call, greek_call.items()):
    col.metric(label=f"Call {name}", value=f"{val:.3f}", border=True)

cols_put = st.columns(4)
for col, (name, val) in zip(cols_put, greek_put.items()):
    col.metric(label=f"Put {name}", value=f"{val:.3f}", border=True)

