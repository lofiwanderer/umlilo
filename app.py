
import streamlit as st
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats

import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import rfft, rfftfreq
import math

# ======================= CONFIG ==========================
st.set_page_config(page_title="CYA Quantum Tracker", layout="wide")
st.title("🔥 CYA MOMENTUM TRACKER: Phase 1 + 2 + 3 + 4")

# ================ SESSION STATE INIT =====================
if "roundsc" not in st.session_state:
    st.session_state.roundsc = []
if "ga_pattern" not in st.session_state:
    st.session_state.ga_pattern = None
if "forecast_msi" not in st.session_state:
    st.session_state.forecast_msi = []

# ================ CONFIGURATION SIDEBAR ==================
with st.sidebar:
    st.header("CONFIGURATION")
    WINDOW_SIZE = st.slider("MSI Window Size", 5, 100, 20)
    PINK_THRESHOLD = st.number_input("Pink Threshold", value=10.0)
    STRICT_RTT = st.checkbox("Strict RTT Mode", value=False)
    if st.button("🔄 Full Reset", help="Clear all historical data"):
        st.session_state.roundsc = []
        st.rerun()
# =================== ROUND ENTRY ========================
st.subheader("Manual Round Entry")
mult = st.number_input("Enter round multiplier", min_value=0.01, step=0.01)

if st.button("➕ Add Round"):
    score = 2 if mult >= PINK_THRESHOLD else (1 if mult >= 2.0 else -1)
    st.session_state.roundsc.append({
        "timestamp": datetime.now(),
        "multiplier": mult,
        "score": score
    })

# =================== CONVERT TO DATAFRAME ================
df = pd.DataFrame(st.session_state.roundsc)

def rrqi(df, window=30):
    recent = df.tail(window)
    blues = len(recent[recent['type'] == 'Blue'])
    purples = len(recent[recent['type'] == 'Purple'])
    pinks = len(recent[recent['type'] == 'Pink'])
    quality = (purples + 2*pinks - blues) / window
    return round(quality, 2)

# === TPI CALCULATIONS ===
def calculate_purple_pressure(df, window=10):
    recent = df.tail(window)
    purple_scores = recent[recent['type'] == 'Purple']['score']
    if len(purple_scores) == 0:
        return 0
    return purple_scores.sum() / window

def calculate_blue_decay(df, window=10):
    recent = df.tail(window)
    blue_scores = recent[recent['type'] == 'Blue']['multiplier']
    if len(blue_scores) == 0:
        return 0
    decay = np.mean([2.0 - b for b in blue_scores])  # The lower the blue, the higher the decay
    return decay * (len(blue_scores) / window)

def compute_tpi(df, window=10):
    pressure = calculate_purple_pressure(df, window)
    decay = calculate_blue_decay(df, window)
    return round(pressure - decay, 2)

def bollinger_bands(series, window, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return rolling_mean, upper_band, lower_band



if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= PINK_THRESHOLD else ("Purple" if x >= 2 else "Blue"))
    df["msi"] = df["score"].rolling(WINDOW_SIZE).sum()
    df["momentum"] = df["score"].cumsum()
    # === Define latest_msi safely ===
    latest_msi = df["msi"].iloc[-1] if not df["msi"].isna().all() else 0
    latest_tpi = compute_tpi(df, window=WINDOW_SIZE)
    
    # Multi-window BBs on MSI
    df["bb_mid_20"], df["bb_upper_20"], df["bb_lower_20"] = bollinger_bands(df["msi"], 20, 2)
    df["bb_mid_10"], df["bb_upper_10"], df["bb_lower_10"] = bollinger_bands(df["msi"], 10, 1.5)
    df["bb_mid_40"], df["bb_upper_40"], df["bb_lower_40"] = bollinger_bands(df["msi"], 40, 2.5)

    df["bb_squeeze"] = df["bb_upper_10"] - df["bb_lower_10"]
    df["bb_squeeze_flag"] = df["bb_squeeze"] < df["bb_squeeze"].rolling(30).quantile(0.25)


    # === Harmonic Cycle Estimation ===
    scores = df['score'].fillna(0).values
    N = len(scores)
    T = 1

    if N >= 20:
        yf = rfft(scores - np.mean(scores))
        xf = rfftfreq(N, T)
        dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
        dominant_cycle = round(1 / dominant_freq) if dominant_freq != 0 else 0
        harmonic_wave = np.sin(2 * np.pi * dominant_freq * np.arange(N))
    else:
        dominant_cycle = None
        harmonic_wave = None

    # === RRQI Calculation ===
    rrqi_val = rrqi(df, 30)

    # ================== MSI CHART =======================
    st.subheader("Momentum Score Index (MSI)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("white")
    # === Zero Axis Line for Orientation ===
    ax.axhline(0, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax.plot(df["timestamp"], df["msi"], color='black', lw=2, label="MSI")

    # MSI Zones
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] >= 6), color='#ff69b4', alpha=0.8, label="Burst Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=((df["msi"] > 3) & (df["msi"] < 6)), color='#00ffff', alpha=0.3, label="Surge Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] <= -3), color='#ff3333', alpha=0.8, label="Pullback Zone")

    # Plot Bollinger Bands
    ax.plot(df["timestamp"], df["bb_upper_20"], color='maroon', linestyle='--', alpha=1.0, label="BB Upper (20)")
    ax.plot(df["timestamp"], df["bb_lower_20"], color='maroon', linestyle='--', alpha=1.0, label="BB Lower (20)")
    ax.plot(df["timestamp"], df["bb_mid_20"], color='maroon', linestyle=':', alpha=1.0)
    
    # Optional: Short-term band
    ax.plot(df["timestamp"], df["bb_upper_10"], color='cyan', linestyle='--', alpha=1.0)
    ax.plot(df["timestamp"], df["bb_lower_10"], color='cyan', linestyle='--', alpha=1.0)

    # Optional: long-term band
    ax.plot(df["timestamp"], df["bb_upper_40"], color='black', linestyle='--', alpha=1.0)
    ax.plot(df["timestamp"], df["bb_lower_40"], color='black', linestyle='--', alpha=1.0)
    

    # Plot squeeze zones
    for i in range(len(df)):
        if df["bb_squeeze_flag"].iloc[i]:
            ax.axvspan(df["timestamp"].iloc[i] - pd.Timedelta(minutes=0.25),
                       df["timestamp"].iloc[i] + pd.Timedelta(minutes=0.25),
                       color='purple', alpha=0.9)

    # RRQI line (optional bubble)
    if rrqi_val:
        ax.axhline(rrqi_val, color='cyan', linestyle=':', alpha=0.9, label='RRQI Level')

    # Harmonic Forecast Plot
    if harmonic_wave is not None:
        ax.plot(df["timestamp"], harmonic_wave, 
                color='green', linestyle='-', alpha=0.7, label=f"Harmonic Cycle ~{dominant_cycle}r")

    ax.set_title("MSI Tactical Map + Harmonics", color='black')
    ax.tick_params(colors='black')
    ax.legend()
    st.pyplot(fig)

    # RRQI Status
    st.metric("🧠 RRQI", rrqi_val, delta="Last 30 rounds")
    if rrqi_val >= 0.3:
        st.success("🔥 Happy Hour Detected — Tactical Entry Zone")
    elif rrqi_val <= -0.2:
        st.error("⚠️ Dead Zone — Avoid Aggressive Entries")
    else:
        st.info("⚖️ Mixed Zone — Scout Cautiously")

    if dominant_cycle is not None:
        if dominant_cycle >= 25:
            st.success(f"🧠 Harmonic Cycle Stable (~{dominant_cycle}r): Conditions Favor Purple/Pink")
        elif 10 <= dominant_cycle < 25:
            st.info(f"⚖️ Harmonic Cycle Moderate (~{dominant_cycle}r): Entry Needs Confirmation")
        else:
            st.error(f"⚠️ Harmonic Cycle Volatile (~{dominant_cycle}r): Increased Blue Risk")

        
    if not df["bb_upper_20"].isna().all():
        future_upper = df["bb_upper_20"].iloc[-1]
        future_lower = df["bb_lower_20"].iloc[-1]
        st.info(f"Forecast MSI Range (Next Rounds): {future_lower:.2f} → {future_upper:.2f}")

    # === TPI INTERPRETATION HUD ===
    st.metric("TPI", f"{latest_tpi}", delta="Trend Pressure")
    
    if latest_msi >= 3:
        if latest_tpi > 0.5:
            st.success("🔥 Valid Surge — Pressure Confirmed")
        elif latest_tpi < -0.5:
            st.warning("⚠️ Hollow Surge — Likely Trap")
        else:
            st.info("🧐 Weak Surge — Monitor Closely")
    else:
        st.info("Trend too soft — TPI not evaluated.")

    
    # Log
    st.subheader("Round Log (Editable)")
    edited = st.data_editor(df.tail(30), use_container_width=True, num_rows="dynamic")
    st.session_state.roundsc = edited.to_dict('records')

else:
    st.info("Enter at least 1 round to begin analysis.")
