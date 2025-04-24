import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ===== SESSION STATE ===== #
if 'pink_zones' not in st.session_state:
    st.session_state.pink_zones = {'multipliers': [], 'indices': []}
if 'momentum_line' not in st.session_state:
    st.session_state.momentum_line = [0]
if 'rounds' not in st.session_state:
    st.session_state.rounds = []
if 'danger_zones' not in st.session_state:
    st.session_state.danger_zones = []
if 'consecutive_blues' not in st.session_state:
    st.session_state.consecutive_blues = 0
if 'last_pink_index' not in st.session_state:
    st.session_state.last_pink_index = -1
if 'entropy_warnings' not in st.session_state:
    st.session_state.entropy_warnings = []
if 'current_pattern' not in st.session_state:
    st.session_state.current_pattern = 'NEUTRAL'

# ===== CORE ENGINE ===== #
def score_round(multiplier):
    """Precision scoring without rounding"""
    if multiplier < 1.5: return -1.5
    return np.interp(multiplier,
                   [1.5, 2.0, 5.0, 10.0, 20.0],
                   [-1.0, 1.0, 1.5, 2.0, 3.0])

def entropy_collapse_detected():
    """Quantum-enhanced entropy collapse detection"""
    if len(st.session_state.rounds) < 8: return False
    recent = st.session_state.rounds[-8:]
    blues = [r for r in recent if r < 2.0]
    if len(blues) >= 5: return True
    if len(blues) >=3:
        trend = np.polyfit(range(len(blues)), [1 if r <2 else 0 for r in blues], 1)[0]
        return trend > 0.25
    return False

def enhanced_fib_trap():
    """Dual-spectrum Fibonacci trap detection"""
    if len(st.session_state.rounds) >=5:
        recent_5 = st.session_state.rounds[-5:]
        if sum(r <2.0 for r in recent_5) >=4: return True
    if len(st.session_state.rounds) >=10:
        window = st.session_state.rounds[-10:]
        blues = [i for i,r in enumerate(window) if r <2.0]
        return len(blues)>=4 and np.mean(blues) >6
    return False

# ===== DYNAMIC SYSTEMS ===== #  
def stop_loss_triggered():
    """Hard stop conditions"""
    if st.session_state.consecutive_blues >=3:
        return True
    if (len(st.session_state.rounds) - st.session_state.last_pink_index) >15:
        return True
    return False

def current_stake():
    """Risk-adjusted stake sizing"""
    pattern = st.session_state.current_pattern  
    base = {  
        'STABLE_FLOW': 0.035,  
        'LUCRATIVE_BURST': 0.05,  
        'RED_EXTENSION': 0.0075,  
        'HOUSE_TRAP': 0.0,  
        'NEUTRAL': 0.02  
    }[pattern]  
    if st.session_state.entropy_warnings: base *= 0.4  
    return min(base, 0.05)

# ===== PATTERN WARFARE ADDITIONS ===== #  
def classify_pattern():  
    patterns = {  
        'STABLE_FLOW': is_stable_flow(),  
        'LUCRATIVE_BURST': is_lucrative_burst(),  
        'RED_EXTENSION': is_red_extension(),  
        'HOUSE_TRAP': detect_house_trap()  
    }  
    return max(patterns, key=patterns.get) if any(patterns.values()) else 'NEUTRAL'

def is_stable_flow():
    pinks = st.session_state.pink_zones['indices']
    if len(pinks) <2: return False
    last_pink = pinks[-1]
    prev_pink = pinks[-2]
    interval = last_pink - prev_pink
    purples = sum(2<=r<10 for r in st.session_state.rounds[prev_pink:last_pink])
    return (interval <=10) and (purples >=3)

def is_lucrative_burst():  
    pinks = st.session_state.pink_zones['indices']  
    return (len(pinks) >=2 and  
            (pinks[-1] - pinks[-2] <=5) and  
            all(r >=2.0 for r in st.session_state.rounds[pinks[-2]+1:pinks[-1]]))  

def is_red_extension():  
    last_pink = st.session_state.last_pink_index  
    if last_pink == -1: return False  
    post_pink = st.session_state.rounds[last_pink+1:]  
    return (sum(r <2.0 for r in post_pink[:8]) >=6 and  
           any(2.0 <= r <3.0 for r in post_pink[8:12]))

def detect_house_trap():
    last_pink = st.session_state.last_pink_index
    if last_pink == -1: return False
    post_pink = st.session_state.rounds[last_pink+1:]
    blues = sum(r <2.0 for r in post_pink[:5])
    fake_purple = any(2<=r<2.5 for r in post_pink[5:8])
    return blues >=4 and fake_purple

def tactical_stake():  
    pattern = st.session_state.current_pattern  
    base = {  
        'STABLE_FLOW': 0.035,  
        'LUCRATIVE_BURST': 0.05,  
        'RED_EXTENSION': 0.0075,  
        'HOUSE_TRAP': 0.0,  
        'NEUTRAL': 0.02  
    }[pattern]  
    if st.session_state.entropy_warnings: base *= 0.4  
    return min(base, 0.05)

def execute_countermeasures():  
    if st.session_state.current_pattern == 'HOUSE_TRAP':  
        st.session_state.danger_zones.append(len(st.session_state.rounds))  
        st.session_state.entropy_warnings.append(len(st.session_state.rounds))  
        return False 
    return True

# ===== TACTICAL VISUALS ===== #
def create_tactical_chart():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Core momentum line
    momentum = pd.Series(st.session_state.momentum_line)
    ax.plot(momentum.ewm(alpha=0.75).mean(),
            color='#00fffa', lw=2, marker='o',
            markersize=8, markerfacecolor='white')

    # Pink reaction zones
    for mult, idx in zip(st.session_state.pink_zones['multipliers'],
                        st.session_state.pink_zones['indices']):
        ax.fill_betweenx(y=[mult*0.95, mult*1.05],
                        x1=0, x2=len(momentum)-1,
                        color='#4a148c', alpha=0.08)
        ax.axvline(idx, color='#ff00ff', linestyle=':', alpha=0.4)

    # Pattern highlights
    current_round = len(st.session_state.rounds)-1
    color_map = {  
        'STABLE_FLOW': '#00ff00',  
        'LUCRATIVE_BURST': '#00ff9d',  
        'RED_EXTENSION': '#ff9100',  
        'HOUSE_TRAP': '#ff0000'  
    }  
    ax.axvspan(current_round-0.5, current_round+0.5,  
              facecolor=color_map.get(st.session_state.current_pattern, '#000000'), 
              alpha=0.15, zorder=0)

    # Danger visualization
    for zone in st.session_state.danger_zones:
        ax.axvspan(zone-0.5, zone+0.5, color='#d50000', alpha=0.15)
    for warning in st.session_state.entropy_warnings:
        ax.axvline(warning, color='#ff9100', alpha=0.3)

    ax.set_title("CYA PATTERN WARFARE v7.0", color='#00fffa', fontsize=18)
    ax.set_facecolor('#000000')
    return fig

# ===== INTERFACE ===== #
st.set_page_config(page_title="CYA Tactical", layout="wide")
st.title("🔥 CYA PATTERN WARFARE")

# Input panel
with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        mult = st.number_input("ENTER MULTIPLIER", 1.0, 1000.0, 1.0, 0.1)
    with col2:
        if st.button("🚀 ANALYZE", type="primary") and not stop_loss_triggered():
            # Core updates
            st.session_state.rounds.append(mult)
            new_score = st.session_state.momentum_line[-1] + score_round(mult)
            st.session_state.momentum_line.append(new_score)
            
            # Track blues
            if mult < 2.0:
                st.session_state.consecutive_blues +=1
            else:
                st.session_state.consecutive_blues =0
            
            # Track pinks
            if mult >=10.0:
                st.session_state.pink_zones['multipliers'].append(mult)
                st.session_state.pink_zones['indices'].append(len(st.session_state.rounds)-1)
                st.session_state.last_pink_index = len(st.session_state.rounds)-1
            
            # Pattern warfare execution
            st.session_state.current_pattern = classify_pattern()
            if not execute_countermeasures():
                st.error("🚫 HOUSE TRAP ACTIVE - BETTING BLOCKED")
                st.session_state.consecutive_blues = 0
            
            # Danger detection
            if enhanced_fib_trap() or entropy_collapse_detected():
                st.session_state.danger_zones.append(len(st.session_state.rounds)-1)
            if entropy_collapse_detected():
                st.session_state.entropy_warnings.append(len(st.session_state.rounds)-1)
            
        if st.button("🔄 FULL RESET", type="secondary"):
            st.session_state.clear()
            st.rerun()

# Main display
st.pyplot(create_tactical_chart())

# Status HUD
with st.container():
    cols = st.columns(5)
    cols[0].metric("TOTAL ROUNDS", len(st.session_state.rounds))
    cols[1].metric("PINK SIGNALS", len(st.session_state.pink_zones['multipliers']))
    cols[2].metric("CURRENT STAKE", f"{tactical_stake()*100:.1f}%")
    cols[3].progress(
        min(100, len(st.session_state.danger_zones)*15),
        text=f"DANGER LEVEL: {len(st.session_state.danger_zones)*15}%"
    )
    cols[4].metric("ACTIVE PATTERN", st.session_state.current_pattern)

# Alert system
if st.session_state.entropy_warnings:
    st.error(f"⚡ ENTROPY COLLAPSE DETECTED ({len(st.session_state.entropy_warnings)} warnings)")
if stop_loss_triggered():
    st.error("🚨 HARD STOP ACTIVATED - CEASE FIRE")
elif st.session_state.danger_zones:
    st.warning(f"⚠️ FIBONACCI TRAP ZONES ({len(st.session_state.danger_zones)})")
