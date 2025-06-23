import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_thre_fused_tdi(df, thre_vals, period=14, signal_period=2):
    prices = df["multiplier"].values

    # Step 1: RSI Calculation
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    #rsi_signal = rsi.rolling(signal_period).mean()
    #upper_band = rsi_signal + bb_range
    #lower_band = rsi_signal - bb_range
    rsi_mid   = rsi.rolling(14).mean()
    rsi_std   = rsi.rolling(14).std()
    upper_band = rsi_mid + 1.2 * rsi_std
    lower_band = rsi_mid - 1.2 * rsi_std
    rsi_signal  = rsi.ewm(span=7, adjust=False).mean()

    # Step 2: THRE slope (inflection detector)
    thre_vals = pd.Series(thre_vals).fillna(method='ffill').fillna(0)
    thre_slope = np.gradient(thre_vals)

    # Step 3: Signal Logic
    entry_arrows = []
    exit_arrows = []
    reversal_arrows = []

    for i in range(len(prices)):
        if i < 2 or i >= len(rsi) or i >= len(thre_vals):
            entry_arrows.append(None)
            exit_arrows.append(None)
            reversal_arrows.append(None)
            continue

        # Values for current step
        rsi_now = rsi.iloc[i]
        rsi_prev = rsi.iloc[i - 1]
        signal_now = rsi_signal.iloc[i]
        signal_prev = rsi_signal.iloc[i - 1]
        thre_now = thre_vals.iloc[i]
        thre_prev = thre_vals.iloc[i - 1]
        slope_now = thre_slope[i]

        # ENTRY: RSI crosses above signal + THRE > 1.2 + THRE rising
        if rsi_prev <= signal_prev and rsi_now > signal_now and thre_now > 1.2 and slope_now > 0:
            entry_arrows.append(rsi_now)
        else:
            entry_arrows.append(None)

        # EXIT: RSI crosses below signal + THRE < -1.2 + THRE falling
        if rsi_prev >= signal_prev and rsi_now < signal_now and thre_now < -1.2 and slope_now < 0:
            exit_arrows.append(rsi_now)
        else:
            exit_arrows.append(None)

        # REVERSAL (Bottom bounce): RSI at/below lower band + THRE rising or >1.5
        if rsi_now <= lower_band.iloc[i] and (thre_now > 1.5 or slope_now > 0):
            reversal_arrows.append(('🟢', rsi_now))  # bounce up
        # REVERSAL (Top pullback): RSI at/above upper band + THRE falling or <1.5
        elif rsi_now >= upper_band.iloc[i] and (thre_now < 1.5 or slope_now < 0):
            reversal_arrows.append(('🔴', rsi_now))  # bounce down
        else:
            reversal_arrows.append(None)

    # Step 4: Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rsi, label='RSI', color='black')
    ax.plot(rsi_signal, label='Signal', color='brown', linestyle='--')
    ax.plot(upper_band, color='gray', linestyle=':', label='Upper Band')
    ax.plot(lower_band, color='red', linestyle=':', label='Lower Band')

    # Entry Arrows 🔼
    for i, val in enumerate(entry_arrows):
        if val is not None:
            ax.annotate('🔼', (i, val), color='lime', fontsize=14, ha='center', va='bottom')

    # Exit Arrows 🔽
    for i, val in enumerate(exit_arrows):
        if val is not None:
            ax.annotate('🔽', (i, val), color='red', fontsize=14, ha='center', va='top')

    # Reversal Arrows (🟢/🔴)
    for i, signal in enumerate(reversal_arrows):
        if signal is not None:
            arrow, val = signal
            ax.annotate(arrow, (i, val), color='white', fontsize=16, ha='center', va='center')

    ax.set_title("🎯 THRE-Fused TDI System with Reversal + Inflection Detection")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)
