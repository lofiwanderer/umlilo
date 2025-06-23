
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_thre_fused_tdi(df, thre_vals, rsi_period=13, signal_period=2):
    if len(df) < rsi_period + signal_period:
        st.warning("Not enough data for TDI + THRE analysis.")
        return

    prices = df['multiplier'].values

    # === TDI Construction ===
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_signal = rsi.rolling(signal_period).mean()
    upper_band = rsi_signal + 5
    lower_band = rsi_signal - 5

    # === THRE Entry/Exit Logic ===
    entry_arrows = []
    exit_arrows = []

    for i in range(len(thre_vals)):
        if i < signal_period or i >= len(rsi):
            entry_arrows.append(None)
            exit_arrows.append(None)
            continue

        thre_val = thre_vals[i]
        is_rsi_cross = rsi.iloc[i] > rsi_signal.iloc[i] and rsi.iloc[i - 1] <= rsi_signal.iloc[i - 1]
        is_rsi_drop = rsi.iloc[i] < rsi_signal.iloc[i] and rsi.iloc[i - 1] >= rsi_signal.iloc[i - 1]

        if thre_val > 1.5 and is_rsi_cross:
            entry_arrows.append(rsi.iloc[i])
        else:
            entry_arrows.append(None)

        if thre_val < -1.2 and is_rsi_drop:
            exit_arrows.append(rsi.iloc[i])
        else:
            exit_arrows.append(None)

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rsi, label='RSI', color='cyan')
    ax.plot(rsi_signal, label='RSI Signal', color='orange', linestyle='--')
    ax.plot(upper_band, color='gray', linestyle=':', label='Upper Band')
    ax.plot(lower_band, color='gray', linestyle=':', label='Lower Band')

    for i in range(len(entry_arrows)):
        if entry_arrows[i] is not None:
            ax.annotate('↑', (i, entry_arrows[i]), color='lime', fontsize=12, ha='center', va='bottom')

    for i in range(len(exit_arrows)):
        if exit_arrows[i] is not None:
            ax.annotate('↓', (i, exit_arrows[i]), color='red', fontsize=12, ha='center', va='top')

    ax.set_title("🔬 THRE-Fused TDI System")
    ax.set_ylabel("RSI")
    ax.legend()
    st.pyplot(fig)
