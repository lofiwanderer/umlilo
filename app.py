import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ======================= CONFIG ==========================
st.set_page_config(page_title="CYA Quantum Tracker", layout="wide")
st.title("🔥 CYA MOMENTUM TRACKER: Phase 1 + 2 + 3")

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

# ================= GA VALIDATOR + FORECAST ENGINE =================
def ga_detect_pattern(msi_series):
    if len(msi_series) < 10:
        return None
    recent = list(msi_series[-10:].fillna(0))
    if recent[-1] < recent[-2] < recent[-3] and recent[-4] > recent[-3]:
        return {"pattern": "Surge Trap", "confidence": 83, "range": (len(msi_series)-10, len(msi_series)-1)}
    if recent[-1] > recent[-2] > recent[-3] and recent[-4] < recent[-3]:
        return {"pattern": "Ramp-Up Burst", "confidence": 74, "range": (len(msi_series)-10, len(msi_series)-1)}
    return None


def forecast_msi(msi_now, avg_score):
    return [round(msi_now + avg_score*(i+1), 2) for i in range(3)]


def get_msi_slope(df, window=3):
        if len(df) < window + 1:
            return 0.0
        y = df["msi"].iloc[-(window+1):].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return round(slope, 2)


# =================== CONVERT TO DATAFRAME ================
df = pd.DataFrame(st.session_state.roundsc)

if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= PINK_THRESHOLD else ("Purple" if x >= 2 else "Blue"))
    df["msi"] = df["score"].rolling(WINDOW_SIZE).sum()
    df["momentum"] = df["score"].cumsum()

    # ======= MDI Calculation =======
    mdi_value = None
    mdi_note = "N/A"
    
    if len(df) >= 6:
        msi_delta = df["msi"].iloc[-1] - df["msi"].iloc[-6]
        mom_delta = df["momentum"].iloc[-1] - df["momentum"].iloc[-6]
    
        if mom_delta != 0:
            mdi_value = round(msi_delta / mom_delta, 2)
            if mdi_value > 1.2:
                mdi_note = "⬆️ Upward Divergence"
            elif mdi_value < -1.2:
                mdi_note = "⬇️ Downward Divergence"
            else:
                mdi_note = "⚖️ Neutral Divergence"


    # PHASE 3 Pattern Validator
    st.session_state.ga_pattern = ga_detect_pattern(df["msi"].fillna(0))

    # PHASE 3 Forecast Bubble
    if len(df) >= WINDOW_SIZE + 3:
        avg_score = np.mean(df["score"].iloc[-WINDOW_SIZE:])
        st.session_state.forecast_msi = forecast_msi(df["msi"].iloc[-1], avg_score)

    # ================== MSI CHART =======================
    st.subheader("Momentum Score Index (MSI)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("black")
    # === Zero Axis Line for Orientation ===
    ax.axhline(0, color='white', linestyle='--', linewidth=3, alpha=0.8)

    ax.plot(df["timestamp"], df["msi"], color='white', lw=2, label="MSI")

    # MSI Zones
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] >= 6), color='#ff69b4', alpha=0.5, label="Burst Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=((df["msi"] > 3) & (df["msi"] < 6)), color='#00ffff', alpha=0.5, label="Surge Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] <= -3), color='#ff3333', alpha=0.5, label="Pullback Zone")

    # Pullback trap detection
    # === Pullback Trap Detection (Corrected)
    pullback_indices = []
    
    for i in range(2, len(df)):
        last_three = df['multiplier'].iloc[i-2:i+1].values
        msi_now = df['msi'].iloc[i]
        
        # Check only if current MSI is in positive territory (≥ 3)
        if msi_now >= 2:
            is_consecutive_blues = all(r < 2.0 for r in last_three)
            is_descending_blues = last_three[0] > last_three[1] > last_three[2] and all(r < 2.0 for r in last_three)
            
            if is_consecutive_blues or is_descending_blues:
                ax.axvspan(df['timestamp'].iloc[i] - pd.Timedelta(minutes=0.5),
                           df['timestamp'].iloc[i] + pd.Timedelta(minutes=0.5),
                           color='red', alpha=0.15)
                pullback_indices.append(i)

    # Pink Projection Zones (round-based, not time-based)
    # Pink Projection Zones: Short (8–12) and Long (18–22)
    pink_idxs = df.index[df['type'] == 'Pink'].tolist()
    for idx in pink_idxs:
        for offset in list(range(8, 13)) + list(range(18, 23)):
            future = idx + offset
            if future < len(df):
                ax.axvspan(df["timestamp"].iloc[future] - pd.Timedelta(minutes=0.25),
                           df["timestamp"].iloc[future] + pd.Timedelta(minutes=0.25),
                           color='magenta', alpha=0.06)


    # MDI
   # if len(df) >= 6:
   #     msi_delta = df["msi"].iloc[-1] - df["msi"].iloc[-6]
   #     mom_delta = df["momentum"].iloc[-1] - df["momentum"].iloc[-6]
   #     if mom_delta != 0:
   #         mdi = msi_delta / mom_delta
   #         if abs(mdi) >= 1.5:
   #             ax.text(df["timestamp"].iloc[-1], df["msi"].max()*0.9,
   #                     f"MDI = {mdi:.2f}", color='yellow', fontsize=10)#
    # Pattern Match Zone
    # === Pattern Match Zone Overlay (Safe Index-Based)
    if st.session_state.ga_pattern:
        pattern_range = st.session_state.ga_pattern['range']
        start_idx = pattern_range[0]
        end_idx = pattern_range[1]

    # Confirm both indices are valid before drawing
        if 0 <= start_idx < len(df) and 0 <= end_idx < len(df):
            start = df["timestamp"].iloc[start_idx]
            end = df["timestamp"].iloc[end_idx]
            ax.axvspan(start, end, color='purple', alpha=0.5)

    

    ax.set_title("MSI Tactical Map", color='white')
    ax.tick_params(colors='white')
    ax.legend()
    st.pyplot(fig)

    # === UI Pullback Trap Warnings
    if pullback_indices:
        st.warning(f"⚠️ {len(pullback_indices)} Pullback Trap(s) Detected in Hot Zones — Watch for Entry Fakes")


    # Log
    st.subheader("Round Log (Editable)")
    edited = st.data_editor(df.tail(30), use_container_width=True, num_rows="dynamic")
    st.session_state.roundsc = edited.to_dict('records')

    # Projections
    st.subheader("Sniper Pink Projections")

    
    # Sniper Projection Detection
    df["projected_by"] = None
    df["projects_to"] = None
    
    for i, row in df.iterrows():
        if row["type"] == "Pink":
            # Check if this pink is projected by a prior pink
            for j, prior in df.iloc[:i].iterrows():
                delta_rounds = i - j
                if prior["type"] == "Pink" and (8 <= delta_rounds <= 12 or 18 <= delta_rounds <= 22):
                    df.at[i, "projected_by"] = prior["timestamp"].strftime("%H:%M:%S")
                    df.at[j, "projects_to"] = df["timestamp"].iloc[i].strftime("%H:%M:%S")
                    break

    st.dataframe(
    df[df["type"] == "Pink"][["timestamp", "multiplier", "projected_by", "projects_to"]].tail(10),
    use_container_width=True
    )
    # Entry Decision
    st.subheader("Entry Decision Assistant")
    latest_msi = df["msi"].iloc[-1]
    # === Visual Slope Display ===
    msi_slope = get_msi_slope(df)
    
    slope_arrow = "↗️" if msi_slope > 0.1 else "↘️" if msi_slope < -0.1 else "➡️"
    slope_color = "green" if msi_slope > 0.1 else "red" if msi_slope < -0.1 else "gray"
    
    st.markdown(f"<span style='color:{slope_color}; font-size: 20px;'>MSI Slope: {slope_arrow} {msi_slope}</span>", unsafe_allow_html=True)

    if latest_msi >= 6:
        st.success("✅ PINK Entry Zone")
    elif 3 <= latest_msi < 6:
        st.info("🟣 PURPLE Surge Opportunity")
    elif latest_msi <= -3:
        st.warning("❌ Pullback Danger — Avoid")
    else:
        st.info("⏳ Neutral — Wait and Observe")

    st.info(f"🧭 MDI: `{mdi_value}` — {mdi_note}")


    # Forecast Display
    if st.session_state.forecast_msi:
        st.subheader("🔮 Forecast MSI Bubble")
        for i, val in enumerate(st.session_state.forecast_msi, 1):
            st.write(f"MSI in {i} rounds: {val}")

    # GA Pattern
    if st.session_state.ga_pattern:
        st.subheader("⚠️ Pattern Warning")
        p = st.session_state.ga_pattern
        st.warning(f"{p['pattern']} Detected — {p['confidence']}% Match")

else:
    st.info("Enter at least 1 round to begin analysis.")
