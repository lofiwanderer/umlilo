import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# === CORE SYSTEM INIT ===
# Replace the existing initialization with this:
if 'rounds' not in st.session_state:
    st.session_state.rounds = []
if 'momentum' not in st.session_state:
    st.session_state.momentum = [0]
if 'pink_zones' not in st.session_state:
    st.session_state.pink_zones = []
if 'fib_traps' not in st.session_state:
    st.session_state.fib_traps = []
if 'entropy_zones' not in st.session_state:
    st.session_state.entropy_zones = []
if 'house_traps' not in st.session_state:
    st.session_state.house_traps = []
if 'minute_clusters' not in st.session_state:
    st.session_state.minute_clusters = []
if 'phase_clusters' not in st.session_state:
    st.session_state.phase_clusters = []
if 'cycle_lengths' not in st.session_state:
    st.session_state.cycle_lengths = []
if 'last_peak' not in st.session_state:
    st.session_state.last_peak = -1
    

# === QUANTUM CORE ===
def score_round(m):
    """Tactical scoring matrix - DO NOT MODIFY"""
    if m < 1.5: return -1.5
    if m < 2.0: return -1.0
    if m < 5.0: return  1.0
    if m < 10.0: return 1.5
    if m < 20.0: return 2.0
    if m < 50.0: return 3.0
    return 4.0  # 50x+

def compute_tactical_ema(momentum, span=3.05, smoothing=1.00):
    """Ultra-responsive EMA for sniper signals"""
    alpha = 2 / (span + 1) * smoothing
    ema = [momentum[0]]
    for price in momentum[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return ema

# === SNIPER DETECTION SUITE ===
def detect_fib_trap(rounds, lookback=8):
    """3+ blues in 5-round window"""
    return [i+4 for i in range(len(rounds)-4) 
           if sum(r < 2.0 for r in rounds[i:i+5]) >=3]

def detect_entropy_collapse(rounds, window=4, threshold=0.48):
    """Volatility collapse detection"""
    return [i for i in range(window, len(rounds)) 
           if np.std(rounds[i-window:i]) < threshold]

def detect_house_trap(rounds):
    """Alternating low/high pattern trap"""
    return [i+2 for i in range(len(rounds)-5)
           if all(r < 2.0 for r in rounds[i:i+3]) and 
           any(2.0 <= r < 2.5 for r in rounds[i+3:i+6])]

def detect_minute_cluster(rounds):
    """:30-:59 blue dominance"""
    return [i for i,r in enumerate(rounds) 
           if (i % 10 >= 5) and (r < 2.0)]

# === NEW PHASE/CYCLE MODULES ===
def update_phase_clusters():
    """Track momentum trios without changing original rounds"""
    if len(st.session_state.momentum) >= 3:
        new_cluster = (
            st.session_state.momentum[-3],
            st.session_state.momentum[-2],
            st.session_state.momentum[-1]
        )
        st.session_state.phase_clusters.append(new_cluster)

def detect_cycles():
    """Peak detection using existing momentum data"""
    momentum = st.session_state.momentum
    if len(momentum) < 5: return
    
    # Detect momentum peaks
    peaks = [i for i in range(1, len(momentum)-1) 
            if momentum[i] > momentum[i-1] 
            and momentum[i] > momentum[i+1]]
    
    # Update cycles only if new peak detected
    if peaks and peaks[-1] > st.session_state.last_peak:
        st.session_state.cycle_lengths.append(peaks[-1] - st.session_state.last_peak)
        st.session_state.last_peak = peaks[-1]

# === TACTICAL VISUALS ===
def plot_sniper_zones(ax, zones, color, label):
    """Precision zone plotting - no merge, full trace"""
    for z in zones:
        ax.axvline(z, color=color, linestyle='--', alpha=0.7)
        ax.axvspan(z-0.3, z+0.3, color=color, alpha=0.15)
    if zones:
        ax.plot([], [], color=color, label=label, marker='s', linestyle='None')

def enhanced_plot(ax):
    """Adds phase/cycle elements to existing plot"""
    # Original momentum plot
    momentum = st.session_state.momentum
    ax.plot(momentum, '-o', color='#00ffff', lw=1.5, 
           markersize=8, markeredgecolor='white', label='Quantum Momentum')
    
    # Original EMA plot
    ema = compute_tactical_ema(momentum)
    ax.plot(ema, '--', color='#ffffff', lw=1.2, alpha=0.9, label='Tactical EMA (3.0)')
    
    # Phase Clusters (2D Projection)
    if st.session_state.phase_clusters:
        x = [c[0] for c in st.session_state.phase_clusters]
        y = [c[1] for c in st.session_state.phase_clusters]
        ax.scatter(x, y, s=30, alpha=0.4, color='#00ff00',
                  label='Phase Clusters')
    
    # Cycle Markers
    for cycle in st.session_state.cycle_lengths:
        ax.axvline(cycle, color='#ffffff', alpha=0.2, 
                  linestyle=':', linewidth=1)
    
    # Original pink zones
    for idx in st.session_state.pink_zones:
        if idx < len(momentum):
            ax.hlines(momentum[idx], 0, len(momentum)-1, 
                     colors='#ff00ff', linestyles=':', alpha=0.4)
            ax.axvline(idx, color='#ff00ff', linestyle='--', alpha=0.6)
    
    # Original sniper zones
    plot_sniper_zones(ax, st.session_state.fib_traps, '#ff0000', 'Fibonacci Trap')
    plot_sniper_zones(ax, st.session_state.entropy_zones, '#ffd700', 'Entropy Collapse')
    plot_sniper_zones(ax, st.session_state.house_traps, '#ffa500', 'House Trap')
    plot_sniper_zones(ax, st.session_state.minute_clusters, '#bf00ff', 'Minute Cluster')

# === CORE SYSTEM INIT ===

# === WAR ROOM INTERFACE ===
st.set_page_config(layout="wide", page_icon="🔥")
st.title("CYBER TACTICAL SNIPER v2.1")

with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        mult = st.number_input("🎯 LIVE MULTIPLIER INPUT", 
                             1.0, 100.0, 1.0, 0.1,
                             key='sniper_input')
    with col2:
        if st.button("🚀 FIRE ANALYSIS", type="primary"):
            # Original core updates
            st.session_state.rounds.append(mult)
            st.session_state.momentum.append(
                st.session_state.momentum[-1] + score_round(mult))
            
            if mult >= 10.0:
                st.session_state.pink_zones.append(len(st.session_state.rounds)-1)
            
            # Original detection suite
            recent_rounds = st.session_state.rounds[-15:] 
            start_idx = max(0, len(st.session_state.rounds) - 15)

            detected_fib = detect_fib_trap(recent_rounds)
            st.session_state.fib_traps += [start_idx + i for i in detected_fib]

            detected_entropy = detect_entropy_collapse(recent_rounds)
            st.session_state.entropy_zones += [start_idx + i for i in detected_entropy]

            detected_house = detect_house_trap(recent_rounds)
            st.session_state.house_traps += [start_idx + i for i in detected_house]

            detected_minute = detect_minute_cluster(recent_rounds)
            st.session_state.minute_clusters += [start_idx + i for i in detected_minute]
            
            # New phase/cycle updates
            update_phase_clusters()
            detect_cycles()
            
        if st.button("☢️ FULL SYSTEM RESET", type="secondary"):
                # PROPER RESET SEQUENCE
                st.session_state.rounds = []
                st.session_state.momentum = [0]
                st.session_state.pink_zones = []
                st.session_state.fib_traps = []
                st.session_state.entropy_zones = []
                st.session_state.house_traps = []
                st.session_state.minute_clusters = []
                st.session_state.phase_clusters = []
                st.session_state.cycle_lengths = []
                st.session_state.last_peak = -1

# === TACTICAL PLOTTER ===
fig, ax = plt.subplots(figsize=(14,7), facecolor='black')
ax.set_facecolor('black')
enhanced_plot(ax)

# Original aesthetics
ax.set_title("CYBER TACTICAL OVERLAY", color='#00ffff', fontsize=16, pad=20)
ax.set_xlabel("Combat Rounds", color='white')
ax.set_ylabel("Momentum Index", color='white')
ax.tick_params(colors='white')
ax.legend(loc='upper left', facecolor='black', edgecolor='cyan')
ax.grid(False)
st.pyplot(fig)

# === TACTICAL HUD ===
with st.expander("LIVE COMBAT TELEMETRY", expanded=True):
    cols = st.columns(6)  # Expanded from 5 to 6 columns
    
    # Original metrics
    cols[0].metric("Active Rounds", len(st.session_state.rounds))
    cols[1].metric("Fibonacci Traps", len(st.session_state.fib_traps), 
                  delta=f"{len(st.session_state.fib_traps[-3:])} new")
    cols[2].metric("Entropy Collapses", len(st.session_state.entropy_zones),
                  delta="⚠️ Critical" if len(st.session_state.entropy_zones)>=3 else "")
    cols[3].metric("House Traps", len(st.session_state.house_traps),
                  delta="🚨 Engaged" if st.session_state.house_traps else "Clear")
    cols[4].metric("Minute Clusters", len(st.session_state.minute_clusters),
                  delta=f"Next: :{55 - (len(st.session_state.rounds)%10)*6:02d}")
    
    # New phase/cycle metric
    cols[5].metric(
        "Phase/Cycle", 
        f"{len(st.session_state.phase_clusters)} Clusters",
        delta=f"Cycle: {st.session_state.cycle_lengths[-1] if st.session_state.cycle_lengths else 'N/A'}"
    )

st.markdown("""
<style>
div.stButton > button:first-child {
    border: 2px solid #00ffff;
    border-radius: 5px;
    color: white;
    background: black;
}
div.stButton > button:hover {
    border: 2px solid #ff00ff;
    background: #111111;
}
</style>
""", unsafe_allow_html=True)
