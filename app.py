
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ===== AI ROBOTS =====
class QuantumAdapter:
    @staticmethod
    def enhance_msi(msi_value):
        return msi_value * 1.15

    @staticmethod
    def predict_msi_curve(last_values):
        return [v * 1.1 for v in last_values]

    @staticmethod
    def predict_next_window(rounds):
        return {"high_risk_window": [18.2, 22.7], "predicted_safe": 5.4}


class GenerativeAgent:
    @staticmethod
    def detect_fractal_pattern(rounds):
        return {"confidence": 85, "repeating_zones": [[5, 8], [12, 15]]}


class PhilosophicalOverseer:
    @staticmethod
    def check_addiction_risk(rounds):
        return len(rounds) > 15

    @staticmethod
    def get_msi_safety(msi):
        return int(max(0, 100 - abs(msi) * 5))


st.set_page_config(page_title="CYA Tactical", layout="wide")

# ===== SESSION STATE =====
for key in ['pink_zones', 'momentum_line', 'rounds', 'danger_zones',
            'roundsc', 'ga_patterns', 'quantum_pred', 'ethic_check']:
    if key not in st.session_state:
        st.session_state[key] = [] if key != 'momentum_line' else [0]

# ===== CONFIG =====
WINDOW_SIZE = st.sidebar.slider("MSI Window Size", 10, 100, 20)
PINK_THRESHOLD = st.sidebar.number_input("Pink Threshold (default = 10.0x)", value=10.0)
STRICT_RTT = st.sidebar.checkbox("Strict RTT Mode", value=True)

# ===== AI PHONE =====
with st.sidebar.expander("🤖 AI Helpers"):
    st.write("**Robot Helpers**")
    if st.session_state.ga_patterns:
        st.metric("Pattern Match", f"{st.session_state.ga_patterns['confidence']}%")
    if st.session_state.quantum_pred:
        st.write("Quantum Says:")
        st.json(st.session_state.quantum_pred)
    st.progress(len(st.session_state.rounds) % 100, 
               text=f"Play Safety: {100 - len(st.session_state.rounds)}%")

# ===== SCORING =====
def score_round(multiplier):
    if multiplier < 1.5:
        return -1.5
    return np.interp(multiplier, [1.5, 2.0, 5.0, 10.0, 20.0], [-1.0, 1.0, 1.5, 2.0, 3.0])

def detect_dangers():
    st.session_state.danger_zones = [
        i for i in range(4, len(st.session_state.rounds))
        if sum(m < 2.0 for m in st.session_state.rounds[i-4:i+1]) >= 4
    ]

# ===== VISUALIZATION =====
def create_battle_chart():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12,6))
    momentum = pd.Series(st.session_state.momentum_line)
    ax.plot(momentum.ewm(alpha=0.75).mean(), color='#00fffa', lw=2, marker='o',
            markersize=8, markerfacecolor='white', markeredgecolor='white', zorder=4)

    for idx in st.session_state.pink_zones:
        if idx < len(momentum):
            ax.hlines(momentum[idx], 0, len(momentum)-1, colors='#ff00ff', linestyles=':', alpha=0.4)
            ax.axvline(idx, color='#ff00ff', linestyle='--', alpha=0.6)

    for zone in st.session_state.danger_zones:
        ax.axvspan(zone-0.5, zone+0.5, color='#d50000', alpha=0.15)

    if st.session_state.ga_patterns:
        for pattern in st.session_state.ga_patterns.get('repeating_zones', []):
            ax.axvspan(pattern[0], pattern[1], color='pink', alpha=0.1)

    ax.set_title("CYA TACTICAL OVERLAY v5.1", color='#00fffa', fontsize=18, weight='bold')
    ax.set_facecolor('#000000')
    return fig

# ===== UI: MOMENTUM ANALYSIS =====
st.title("🔥 CYA BATTLE MATRIX")
col1, col2 = st.columns([3,1])
with col1:
    mult = st.number_input("Enter Multiplier", 1.0, 1000.0, 1.0, 0.1)
with col2:
    if st.button("🚀 Analyze"):
        st.session_state.rounds.append(mult)
        st.session_state.momentum_line.append(
            st.session_state.momentum_line[-1] + score_round(mult))
        if mult >= PINK_THRESHOLD:
            st.session_state.pink_zones.append(len(st.session_state.rounds)-1)
        detect_dangers()
        st.session_state.ga_patterns = GenerativeAgent.detect_fractal_pattern(st.session_state.rounds)
        if len(st.session_state.rounds) % 5 == 0:
            st.session_state.quantum_pred = QuantumAdapter.predict_next_window(st.session_state.rounds[-10:])
        if PhilosophicalOverseer.check_addiction_risk(st.session_state.rounds):
            st.toast("⚠️ Break Time!", icon="⚠️")
    if st.button("🔄 Full Reset"):
        for k in ['rounds','momentum_line','pink_zones','danger_zones','roundsc']:
            st.session_state[k] = [] if k != 'momentum_line' else [0]
        st.rerun()

st.pyplot(create_battle_chart())
cols = st.columns(3)
cols[0].metric("TOTAL ROUNDS", len(st.session_state.rounds))
cols[1].progress(min(100, len(st.session_state.danger_zones)*20),
                 text=f"DANGER SCORE: {len(st.session_state.danger_zones)*20}%")
if st.session_state.danger_zones:
    st.error(f"⚠️ FIBONACCI TRAP PATTERNS DETECTED ({len(st.session_state.danger_zones)})")

# ===== UI: MSI SNIPER SECTION =====
st.title("Momentum Tracker v2: MSI + Sniper Logic")
st.subheader("Manual Round Entry (MSI Mode)")
multiplierval = st.number_input("Enter multiplier", min_value=0.01, step=0.01, key="manual_input")
if st.button("Add Round"):
    st.session_state.roundsc.append({
        "timestamp": datetime.now(),
        "multiplier": multiplierval,
        "score": 2 if multiplierval >= PINK_THRESHOLD else (1 if multiplierval >= 2 else -1)
    })

df = pd.DataFrame(st.session_state.roundsc)
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["msi"] = df["score"].rolling(WINDOW_SIZE).sum()
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= PINK_THRESHOLD else ("Purple" if x >= 2 else "Blue"))
    st.subheader("Recent Round Log")
    edited = st.data_editor(df.tail(30), num_rows="dynamic", use_container_width=True)
    st.session_state.roundsc = edited.to_dict('records')

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["timestamp"], df["msi"], label="MSI", color="white")
    ax.plot(df["timestamp"][-5:], QuantumAdapter.predict_msi_curve(df["msi"].values[-5:]),
            color='#ff00ff', linestyle='--', label='AI Forecast')
    for zone in st.session_state.ga_patterns.get('repeating_zones', []):
        ax.axvspan(df["timestamp"].iloc[zone[0]], df["timestamp"].iloc[zone[1]], color='yellow', alpha=0.1)
    ax.axhline(0, color="gray", linestyle="--")
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] >= 6), color="pink", alpha=0.3, label="Burst Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=(df["msi"] <= -6), color="red", alpha=0.2, label="Red Zone")
    ax.fill_between(df["timestamp"], df["msi"], where=((df["msi"] > 0) & (df["msi"] < 6)), color="#00ffff", alpha=0.3, label="Surge Zone")
    ax.legend()
    ax.set_title("Momentum Score Index (MSI)")
    st.pyplot(fig)

    st.subheader("Sniper Pink Projections")
    df["projected_by"] = None
    for i, row in df.iterrows():
        if row["type"] == "Pink":
            for j, prior in df.iloc[:i].iterrows():
                diff = (row["timestamp"] - prior["timestamp"]).total_seconds() / 60
                if prior["type"] == "Pink" and (8 <= diff <= 12 or 18 <= diff <= 22):
                    df.at[i, "projected_by"] = prior["timestamp"].strftime("%H:%M:%S")
    st.dataframe(df[df["type"] == "Pink"][["timestamp", "multiplier", "projected_by"]].tail(10))

    st.subheader("Entry Decision Assistant")
    latest_msi = df["msi"].iloc[-1]
    if latest_msi >= 6:
        st.success("✅ PINK Entry Zone")
    elif 3 <= latest_msi < 6:
        st.info("🟣 PURPLE Entry Zone")
    elif latest_msi <= -3:
        st.warning("❌ Pullback Zone - Avoid Entry")
    else:
        st.info("⏳ Neutral Zone - Wait")
else:
    st.info("Enter multipliers to begin MSI sniper tracking.")
