import streamlit as st
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import sklearn
import pywt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert
import math
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import gridspec
import morlet_phase_enhancement
from morlet_phase_enhancement import morlet_phase_panel

# ======================= CONFIG ==========================
st.set_page_config(page_title="CYA Quantum Tracker", layout="wide")
st.title("🔥 CYA MOMENTUM TRACKER: Phase 1 + 2 + 3 + 4")

# Create a container for the floating add round button
floating_container = st.empty()
latest_rds = None
latest_delta = None

# ================ SESSION STATE INIT =====================
if "roundsc" not in st.session_state:
    st.session_state.roundsc = []
if "ga_pattern" not in st.session_state:
    st.session_state.ga_pattern = None
if "forecast_msi" not in st.session_state:
    st.session_state.forecast_msi = []
if "completed_cycles" not in st.session_state:
    st.session_state.completed_cycles = 0
if "last_position" not in st.session_state:
    st.session_state.last_position = 0
if "current_mult" not in st.session_state:
    st.session_state.current_mult = 2.0

# ================ CONFIGURATION SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ QUANTUM PARAMETERS")
    WINDOW_SIZE = st.slider("MSI Window Size", 5, 100, 20)
    PINK_THRESHOLD = st.number_input("Pink Threshold", value=10.0)
    STRICT_RTT = st.checkbox("Strict RTT Mode", value=False)

    st.header("📊 PANEL TOGGLES")
    FAST_ENTRY_MODE = st.checkbox("⚡ Fast Entry Mode", value=False)
    show_thre = st.checkbox("🌀 THRE Panel", value=True)
    show_cos_panel = st.checkbox("🌀 Cos Phase Panel", value=True)
    show_rqcf = st.checkbox("🔮 RQCF Panel", value=True)
    show_fpm = st.checkbox("🧬 FPM Panel", value=True)
    show_anchor = st.checkbox("🔗 Fractal Anchor", value=True)
    
    if st.button("🔄 Full Reset", help="Clear all historical data"):
        st.session_state.roundsc = []
        st.rerun()
        
    # Clear cached functions
    if st.button("🧹 Clear Cache", help="Force harmonic + MSI recalculation"):
        st.cache_data.clear()  # Streamlit's built-in cache clearer
        st.success("Cache cleared — recalculations will run fresh.")

# =================== ADVANCED HELPER FUNCTIONS ========================
@st.cache_data
def bollinger_bands(series, window, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return rolling_mean, upper_band, lower_band

@st.cache_data
def detect_dominant_cycle(scores):
    N = len(scores)
    if N < 20:
        return None
    T = 1
    yf = rfft(scores - np.mean(scores))
    xf = rfftfreq(N, T)
    dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
    if dominant_freq == 0:
        return None
    return round(1 / dominant_freq)

@st.cache_data
def get_phase_label(position, cycle_length):
    pct = (position / cycle_length) * 100
    if pct <= 16:
        return "Birth Phase", pct
    elif pct <= 33:
        return "Ascent Phase", pct
    elif pct <= 50:
        return "Peak Phase", pct
    elif pct <= 67:
        return "Post-Peak", pct
    elif pct <= 84:
        return "Falling Phase", pct
    else:
        return "End Phase", pct

@st.cache_data
def morlet_wavelet_power(scores, wavelet='cmor', max_scale=64):
    scales = np.arange(1, max_scale)
    
    # Compute CWT
    coeffs, freqs = pywt.cwt(scores, scales, wavelet)
    power = np.abs(coeffs) ** 2
    
    return coeffs, power, freqs, scales
    
@st.cache_data
def classify_morlet_bursts(mean_energy, power_threshold=None, min_width=3):
    if power_threshold is None:
        power_threshold = np.mean(mean_energy) + np.std(mean_energy)

    # Detect peaks
    peaks, props = find_peaks(mean_energy, height=power_threshold)

    # Measure widths
    widths, h_eval, left_ips, right_ips = peak_widths(mean_energy, peaks, rel_height=0.5)

    classifications = []
    for i, peak in enumerate(peaks):
        width = widths[i]
        label = "💥 SURGE PHASE" if width >= min_width else "⚠️ FAKE BURST"
        classifications.append({
            "index": peak,
            "width": round(width, 2),
            "strength": round(props['peak_heights'][i], 2),
            "label": label
        })
    
    return classifications
    
@st.cache_data
def compute_fnr_index_from_morlet(power_matrix, scales):
    if power_matrix.shape[0] < 10:
        return None

    # Extract high-scale (macro) and low-scale (micro) energy patterns
    micro_band = power_matrix[2:6, :]     # e.g., scales 3–6
    macro_band = power_matrix[-6:-2, :]   # e.g., top 4 macro scales

    micro_wave = np.mean(micro_band, axis=0)
    macro_wave = np.mean(macro_band, axis=0)

    # Normalize both
    micro_wave -= np.mean(micro_wave)
    macro_wave -= np.mean(macro_wave)
    micro_wave /= (np.std(micro_wave) + 1e-9)
    macro_wave /= (np.std(macro_wave) + 1e-9)

    # Product method (constructive > 0, destructive < 0)
    interaction_wave = micro_wave * macro_wave
    fnr_index = np.mean(interaction_wave)

    # Cosine similarity (phase match strength)
    cos_sim = cosine_similarity(micro_wave.reshape(1, -1), macro_wave.reshape(1, -1))[0][0]

    return {
        "FNR_index": round(fnr_index, 4),
        "Phase_cosine_similarity": round(cos_sim, 4),
        "alignment": "Constructive" if fnr_index > 0.2 else "Destructive" if fnr_index < -0.2 else "Neutral"
    }
    
@st.cache_data
def calculate_purple_pressure(df, window=10):
    recent = df.tail(window)
    purple_scores = recent[recent['type'] == 'Purple']['score']
    if len(purple_scores) == 0:
        return 0
    return purple_scores.sum() / window

@st.cache_data
def calculate_blue_decay(df, window=10):
    recent = df.tail(window)
    blue_scores = recent[recent['type'] == 'Blue']['multiplier']
    if len(blue_scores) == 0:
        return 0
    decay = np.mean([2.0 - b for b in blue_scores])  # The lower the blue, the higher the decay
    return decay * (len(blue_scores) / window)

@st.cache_data
def compute_tpi(df, window=10):
    pressure = calculate_purple_pressure(df, window)
    decay = calculate_blue_decay(df, window)
    return round(pressure - decay, 2)

@st.cache_data
def rrqi(df, window=30):
    recent = df.tail(window)
    blues = len(recent[recent['type'] == 'Blue'])
    purples = len(recent[recent['type'] == 'Purple'])
    pinks = len(recent[recent['type'] == 'Pink'])
    quality = (purples + 2*pinks - blues) / window
    return round(quality, 2)

@st.cache_data
def multi_harmonic_resonance_analysis(df, num_harmonics=5):
    scores = df["score"].fillna(0).values
    N = len(scores)
    yf = rfft(scores - np.mean(scores))
    xf = rfftfreq(N, 1)
    amplitudes = np.abs(yf)
    top_indices = amplitudes.argsort()[-num_harmonics:][::-1] if len(amplitudes) >= num_harmonics else amplitudes.argsort()
    resonance_matrix = np.zeros((num_harmonics, num_harmonics))
    harmonic_waves = []
    
    for i, idx in enumerate(top_indices):
        if i < len(xf) and i < len(yf):
            freq = xf[idx]
            phase = np.angle(yf[idx])
            wave = np.sin(2 * np.pi * freq * np.arange(N) + phase)
            harmonic_waves.append(wave)
            for j, jdx in enumerate(top_indices):
                if i != j and j < len(yf):
                    phase_diff = np.abs(phase - np.angle(yf[jdx]))
                    resonance_matrix[i,j] = np.cos(phase_diff) * min(amplitudes[idx], amplitudes[jdx])

    resonance_score = np.sum(resonance_matrix) / (num_harmonics * (num_harmonics - 1)) if num_harmonics > 1 else 0
    tension = np.var(amplitudes[top_indices]) if len(top_indices) > 0 else 0
    harmonic_entropy = stats.entropy(amplitudes[top_indices] / np.sum(amplitudes[top_indices])) if len(top_indices) > 0 and np.sum(amplitudes[top_indices]) > 0 else 0
    return harmonic_waves, resonance_matrix, resonance_score, tension, harmonic_entropy

@st.cache_data
def resonance_forecast(harmonic_waves, resonance_matrix, steps=10):
    if not harmonic_waves: return np.zeros(steps)
    forecast = np.zeros(steps)
    num_harmonics = len(harmonic_waves)
    
    for step in range(steps):
        step_value = 0
        for i in range(num_harmonics):
            wave = harmonic_waves[i]
            if len(wave) > 1:
                diff = np.diff(wave[1:])
                freq = 1 / (np.argmax(diff) + 1) if np.any(diff) else 1
                next_val = wave[-1] * np.cos(2 * np.pi * freq * 1)
                influence = np.sum(resonance_matrix[i]) / (num_harmonics - 1) if num_harmonics > 1 else 0
                step_value += next_val * (1 + influence)
                harmonic_waves[i] = np.append(wave, step_value)
        forecast[step] = step_value / num_harmonics
    return forecast

@st.cache_data
def run_rqcf(scores, steps=3, top_n=5):
    if len(scores) < 10: return []
    
    N = len(scores)
    yf = rfft(scores - np.mean(scores))
    xf = rfftfreq(N, 1)
    amplitudes = np.abs(yf)
    top_indices = amplitudes.argsort()[-top_n:][::-1] if len(amplitudes) >= top_n else amplitudes.argsort()
    harmonic_data = []
    
    for idx in top_indices:
        if idx < len(xf) and idx < len(yf) and idx < len(amplitudes):
            freq = xf[idx]
            phase = np.angle(yf[idx])
            amp = amplitudes[idx]
            wave = np.sin(2 * np.pi * freq * np.arange(N) + phase)
            harmonic_data.append((freq, phase, amp, wave))

    forecast_chains = []
    for branch_id in range(3):
        chain = []
        sim_scores = list(scores)
        for step in range(steps):
            wave_sum = np.zeros(1)
            for freq, phase, amp, _ in harmonic_data:
                t = len(sim_scores)
                value = amp * np.sin(2 * np.pi * freq * t + phase)
                wave_sum += value
            score_estimate = wave_sum[0] / max(1, len(harmonic_data))
            sim_scores.append(score_estimate)
            label = '💖 Pink Spike' if score_estimate >= 1.5 else \
                    '🟣 Purple Stable' if score_estimate >= 0.5 else \
                    '🔵 Blue Pullback' if score_estimate < 0 else '⚪ Neutral Drift'
            chain.append((round(score_estimate, 3), label))
            for i in range(len(harmonic_data)):
                freq, phase, amp, wave = harmonic_data[i]
                harmonic_data[i] = (freq, phase + np.random.uniform(-0.1, 0.1), amp, wave)
        forecast_chains.append({"branch": f"Branch {chr(65 + branch_id)}", "forecast": chain})
    return forecast_chains

# =================== UI COMPONENTS ========================
def decision_hud_panel(dominant_phase, dominant_pct, micro_phase, micro_pct,
                       resonance_score, fractal_match_type=None, anchor_forecast_type=None):
    score = 0
    reasons = []
    
    if dominant_phase in ["Ascent Phase", "Peak Phase"]:
        score += 1
        reasons.append("✅ Dominant in profit zone")
    if micro_phase == dominant_phase:
        score += 1
        reasons.append("✅ Micro matches Dominant")
    if resonance_score is not None:
        if resonance_score > 0.7:
            score += 1
            reasons.append("✅ Coherence High")
        elif resonance_score < 0.4:
            score -= 1
            reasons.append("⚠️ Coherence Low")
    if fractal_match_type == "Pink":
        score += 2
        reasons.append("🔥 Fractal Pulse → Pink")
    elif fractal_match_type == "Purple":
        score += 1
        reasons.append("🟣 Fractal Pulse → Purple")
    elif fractal_match_type == "Blue":
        score -= 1
        reasons.append("🔵 Fractal Pulse → Blue")
    if anchor_forecast_type == "Pink":
        score += 2
        reasons.append("💥 Fractal Anchor → Pink")
    elif anchor_forecast_type == "Purple":
        score += 1
        reasons.append("🟪 Fractal Anchor → Purple")
    elif anchor_forecast_type == "Blue":
        score -= 1
        reasons.append("🧊 Fractal Anchor → Blue")
    
    if score >= 4:
        banner_color = "🟢 ENTRY CONFIRMED"
        status = "💥 High Probability Surge"
    elif score >= 2:
        banner_color = "🟡 SCOUT ZONE"
        status = "🧘‍♂️ Wait for Confirmation"
    else:
        banner_color = "🔴 HOLD FIRE"
        status = "⚠️ Likely Trap or Blue Run"
    
    with st.container():
        st.markdown("---")
        st.markdown("### 🎯 **Real-Time Entry Signal HUD**")
        st.markdown(f"**{banner_color}** — {status}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dominant Phase", f"{dominant_phase} ({dominant_pct:.1f}%)")
            st.metric("Microwave Phase", f"{micro_phase} ({micro_pct:.1f}%)")
        with col2:
            st.metric("Fractal Pulse Match", fractal_match_type or "N/A")
            st.metric("Anchor Forecast", anchor_forecast_type or "N/A")
        with col3:
            if resonance_score is not None:
                st.metric("Resonance Score", f"{resonance_score:.2f}")
            else:
                st.metric("Resonance", "N/A")
            st.metric("Signal Score", f"{score} pts")
        with st.expander("🧠 Signal Breakdown"):
            for reason in reasons: st.markdown(f"- {reason}")
        st.markdown("---")

def string_metrics_panel(tension, entropy, resonance_score):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("String Tension", f"{tension:.4f}", 
                 help="Variance in harmonic amplitudes - higher tension indicates unstable state")
    
    with col2:
        st.metric("Harmonic Entropy", f"{entropy:.4f}",
                 help="Information entropy of harmonic distribution - lower entropy predicts stability")
    
    with col3:
        st.metric("Resonance Coherence", f"{resonance_score:.4f}",
                 help="Phase alignment between harmonics - higher coherence predicts constructive interference")

def classify_next_round(forecast, tension, entropy, resonance_score):
    if forecast is None or len(forecast) == 0:
        return "❓ Unknown", "⚠️ No forecast", 0
    
    energy_index = np.tanh(forecast[0])
    classification = "❓ Unknown"
    
    if energy_index > 0.8 and tension < 0.2 and entropy < 1.5:
        classification = "💖 Pink Surge Expected"
    elif energy_index > 0.4:
        classification = "🟣 Probable Purple Round"
    elif -0.4 <= energy_index <= 0.4:
        classification = "⚪ Neutral Drift Zone"
    elif energy_index < -0.8 and tension < 0.15:
        classification = "⚠️ Collapse Risk (Blue Train)"
    elif energy_index < -0.4:
        classification = "🔵 Likely Blue / Pullback"

    if resonance_score > 0.7:
        if energy_index > 0.8: action = "🔫 Sniper Entry — Surge Incoming"
        elif energy_index < -0.8: action = "❌ Abort Entry — Blue Collapse"
        else: action = "🧭 Cautious Scout — Mild Fluctuation"
    else:
        action = "⚠️ Unstable Harmonics — Avoid Entry"

    return classification, action, energy_index

def thre_panel(df):
    st.subheader("🔬 True Harmonic Resonance Engine (THRE)")
    if len(df) < 20: 
        st.warning("Need at least 20 rounds to compute THRE.")
        return df, None, None
        
    scores = df["score"].fillna(0).values
    N = len(scores)
    T = 1
    yf = rfft(scores - np.mean(scores))
    xf = rfftfreq(N, T)
    mask = (xf > 0) & (xf < 0.5)
    freqs = xf[mask]
    amps = np.abs(yf[mask])
    phases = np.angle(yf[mask])
    harmonic_matrix = np.zeros((N, len(freqs)))
    
    for i, (f, p) in enumerate(zip(freqs, phases)):
        harmonic_matrix[:, i] = np.sin(2 * np.pi * f * np.arange(N) + p)
    
    composite_signal = (harmonic_matrix * amps).sum(axis=1) if amps.size > 0 else np.zeros(N)
    normalized_signal = (composite_signal - np.mean(composite_signal)) / np.std(composite_signal) if np.std(composite_signal) > 0 else np.zeros(N)
    smooth_rds = pd.Series(normalized_signal).rolling(3, min_periods=1).mean()
    rds_delta = np.gradient(smooth_rds)
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(df["timestamp"], smooth_rds, label="THRE Resonance", color='cyan')
    ax[0].axhline(1.5, linestyle='--', color='green', alpha=0.5)
    ax[0].axhline(0.5, linestyle='--', color='blue', alpha=0.3)
    ax[0].axhline(-0.5, linestyle='--', color='orange', alpha=0.3)
    ax[0].axhline(-1.5, linestyle='--', color='red', alpha=0.5)
    ax[0].set_title("Composite Harmonic Resonance Strength")
    ax[0].legend()
    
    ax[1].plot(df["timestamp"], rds_delta, label="Δ Resonance Slope", color='purple')
    ax[1].axhline(0, linestyle=':', color='gray')
    ax[1].set_title("RDS Inflection Detector")
    ax[1].legend()
    
    st.pyplot(fig)
    
    latest_rds = smooth_rds.iloc[-1] if len(smooth_rds) > 0 else 0
    latest_delta = rds_delta[-1] if len(rds_delta) > 0 else 0
    
    st.metric("🧠 Resonance Strength", f"{latest_rds:.3f}")
    st.metric("📉 Δ Slope", f"{latest_delta:.3f}")
    
    if latest_rds > 1.5: st.success("💥 High Constructive Stack — Pink Burst Risk ↑")
    elif latest_rds > 0.5: st.info("🟣 Purple Zone — Harmonically Supported")
    elif latest_rds < -1.5: st.error("🌪️ Collapse Zone — Blue Train Likely")
    elif latest_rds < -0.5: st.warning("⚠️ Destructive Micro-Waves — High Risk")
    else: st.info("⚖️ Neutral Zone — Mid-Range Expected")
    
    return (df, latest_rds, latest_delta)
    
def compute_surge_probability(thre_val, delta_slope, fnr_index):
    # Normalize inputs
    thre_score = np.clip((thre_val + 2) / 4, 0, 1)          # maps -2→1 to 0→1
    slope_score = np.clip((delta_slope + 1) / 2, 0, 1)       # maps -1→1 to 0→1
    fnr_score = np.clip((fnr_index + 1) / 2, 0, 1)           # maps -1→1 to 0→1

    # Weighted blend (adjustable)
    surge_prob = 0.5 * thre_score + 0.3 * fnr_score + 0.2 * slope_score
    return round(surge_prob, 4), {
        "thre_component": round(thre_score, 4),
        "fnr_component": round(fnr_score, 4),
        "slope_component": round(slope_score, 4)
    }

def cos_phase_panel(df, dom_freq, micro_freq, dom_phase, micro_phase):
    st.subheader("🌀 Cosine Phase Alignment Panel")
    if df is None or len(df) < 20:
        st.warning("Need at least 20 rounds to compute Phase alignment.")
        return
        
    scores = df["score"].fillna(0).values
    N = len(scores)
    timestamps = df["timestamp"]

    if N >= 20 and dom_freq > 0 and micro_freq > 0:
        # Compute current waveforms
        t = np.arange(N)
        dom_wave = np.sin(2 * np.pi * dom_freq * t + dom_phase)
        micro_wave = np.sin(2 * np.pi * micro_freq * t + micro_phase)

        phase_diff = 2 * np.pi * (dom_freq - micro_freq) * np.arange(N) + (dom_phase - micro_phase)
        alignment_score = np.cos(phase_diff)
        smoothed_score = pd.Series(alignment_score).rolling(5, min_periods=1).mean()

        # Forecast alignment (next 10 rounds)
        forecast_len = 10
        future_t = np.arange(N, N + forecast_len)
        future_align = np.cos(2 * np.pi * (dom_freq - micro_freq) * future_t + (dom_phase - micro_phase))
        forecast_times = [timestamps.iloc[-1] + pd.Timedelta(seconds=5 * i) for i in range(forecast_len)]

        # === Plotting ===
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Past wave alignment
        ax[0].plot(timestamps, dom_wave, label="Dominant Wave", color='blue')
        ax[0].plot(timestamps, micro_wave, label="Micro Wave", color='green', linestyle='dashdot')
        ax[0].set_title("Dominant vs Micro Harmonics")
        ax[0].legend()

        # Cosine phase alignment tracker
        ax[1].plot(timestamps, alignment_score, label="Cos(Δϕ)", color='purple')
        ax[1].plot(timestamps, smoothed_score, linestyle='--', label="Cos(Δϕ)Smooth", color='purple')

        ax[1].axhline(0.5, linestyle='--', color='gray')
        ax[1].axhline(-0.5, linestyle='--', color='gray')
        ax[1].set_title("Cosine Phase Alignment Oscillator")
        ax[1].legend()

        plot_slot = st.empty()
        with plot_slot.container():
            st.pyplot(fig)

        # === Decision HUD ===
        st.subheader("🎯 Phase Decision HUD")
        recent_avg = np.mean(alignment_score[-5:])
        st.metric("Avg Alignment (Last 5)", round(recent_avg, 3))

        if recent_avg > 0.7:
            st.success("ENTRY SIGNAL: Strong Constructive Interference")
        elif recent_avg < -0.7:
            st.error("NO ENTRY: Strong Destructive Interference")
        else:
            st.warning("NEUTRAL FIELD: Proceed with Caution")

    else:
        st.info("⛔ Not enough data or wave definition to compute phase alignment.")

def fpm_panel(df, msi_col="msi", score_col="score", window_sizes=[5, 8, 13]):
    st.subheader("🧬 Fractal Pulse Matcher Panel (FPM)")

    if len(df) < max(window_sizes) + 5:
        st.warning("Not enough historical rounds to match fractal sequences.")
        return

    df = df.copy()
    df["round_type"] = df["score"].apply(lambda s: "P" if s == 2 else ("p" if s == 1 else "B"))

    for win in window_sizes:
        current_seq = df.tail(win).reset_index(drop=True)

        # Encode current pattern
        current_pattern = current_seq["round_type"].tolist()
        current_slope = np.gradient(current_seq[msi_col].fillna(0).values)
        current_fft = np.abs(rfft(current_slope)) if len(current_slope) > 0 else np.array([])

        best_match = None
        best_score = -np.inf
        matched_seq = None
        next_outcome = None

        # Slide through history
        for i in range(0, len(df) - win - 3):
            hist_seq = df.iloc[i:i+win]
            hist_pattern = hist_seq["round_type"].tolist()
            hist_slope = np.gradient(hist_seq[msi_col].fillna(0).values)
            hist_fft = np.abs(rfft(hist_slope)) if len(hist_slope) > 0 else np.array([])

            # Compare slope shape using cosine similarity if both FFTs have data
            sim_score = 0
            if len(current_fft) > 0 and len(hist_fft) > 0:
                # Check if the lengths match; if not, use the smaller length for both
                min_len = min(len(current_fft), len(hist_fft))
                if min_len > 0:
                    sim_score = cosine_similarity([current_fft[:min_len]], [hist_fft[:min_len]])[0][0]

            # Compare round pattern similarity
            pattern_match = sum([a == b for a, b in zip(current_pattern, hist_pattern)]) / win

            # Combined matching score
            total_score = 0.6 * sim_score + 0.4 * pattern_match

            if total_score > best_score:
                best_score = total_score
                best_match = hist_pattern
                matched_seq = hist_seq
                # Look at what happened next
                if i + win + 3 <= len(df):
                    next_seq = df.iloc[i+win:i+win+3]
                    next_outcome = next_seq["round_type"].tolist() if len(next_seq) > 0 else []

        # === Display Results ===
        st.markdown(f"Fractal Match: Last {win} Rounds")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Current Pattern (Last {win}):**")
            st.text(" ".join(current_pattern))
            st.markdown(f"**MSI Slope:** {np.round(current_slope, 2)}")

        with col2:
            st.markdown(f"**Best Historical Match:**")
            st.text(" ".join(best_match) if best_match else "N/A")
            st.markdown(f"**Match Score:** {best_score:.3f}")

        if next_outcome and len(next_outcome) > 0:
            st.success(f"📡 Projected Next Rounds: {' '.join(next_outcome)}")
            # Simple forecast classifier
            if next_outcome.count("P") + next_outcome.count("p") >= 2:
                st.markdown("🔮 Forecast: **💥 Surge Mirror**")
                st.session_state.last_fractal_match = "Pink"
            elif next_outcome.count("B") >= 2:
                st.markdown("⚠️ Forecast: **Blue Reversal / Collapse**")
                st.session_state.last_fractal_match = "Blue"
            else:
                st.markdown("🧘 Forecast: **Stable / Mixed Pulse**")
                st.session_state.last_fractal_match = "Purple"
        else:
            st.session_state.last_fractal_match = None

def fractal_anchor_visualizer(df, msi_col="msi", score_col="score", window=8):
    st.subheader("🔗 Fractal Anchoring Visualizer")

    if len(df) < window + 10:
        st.warning("Insufficient data for visual fractal anchoring.")
        return

    df = df.copy()
    df["type"] = df["score"].apply(lambda s: "P" if s == 2 else ("p" if s == 1 else "B"))

    # Encode recent fragment
    recent_seq = df.tail(window)
    recent_vec = recent_seq[msi_col].fillna(0).values
    recent_types = recent_seq["type"].tolist()

    best_score = -np.inf
    best_start = None
    best_future_types = []

    for i in range(len(df) - window - 3):
        hist_seq = df.iloc[i:i+window]
        hist_vec = hist_seq[msi_col].fillna(0).values
        hist_types = hist_seq["type"].tolist()

        if len(hist_vec) != window:
            continue

        # Cosine similarity between shapes
        sim_score = 0
        if len(recent_vec) > 0 and len(hist_vec) > 0:
            sim_score = cosine_similarity([recent_vec], [hist_vec])[0][0]

        type_match = sum([a == b for a, b in zip(hist_types, recent_types)]) / window
        total_score = 0.6 * sim_score + 0.4 * type_match

        if total_score > best_score:
            best_score = total_score
            best_start = i
            if i + window + 3 <= len(df):
                best_future_types = df.iloc[i+window:i+window+3]["type"].tolist()

    if best_start is None:
        st.warning("No matching historical pattern found.")
        return

    # === Prepare plot ===
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # Historical pattern
    hist_fragment = df.iloc[best_start:best_start+window]
    hist_times = np.arange(-window, 0)
    hist_vals = hist_fragment[msi_col].fillna(0).values
    hist_types = hist_fragment["type"].tolist()
    ax.plot(hist_times, hist_vals, color='gray', linewidth=2, label='Matched Past')

    # Current pattern
    curr_vals = recent_seq[msi_col].fillna(0).values
    ax.plot(hist_times, curr_vals, color='blue', linewidth=2, linestyle='--', label='Current')

    # Forecast next steps
    if best_start + window + 3 <= len(df):
        proj_seq = df.iloc[best_start + window : best_start + window + 3]
        proj_vals = proj_seq[msi_col].fillna(0).values
        proj_times = np.arange(1, len(proj_vals)+1)
        ax.plot(proj_times, proj_vals, color='green', linewidth=2, label='Projected Next')

        # Round type markers
        for t, y in zip(proj_times, proj_vals):
            ax.scatter(t, y, s=100, alpha=0.7,
                       c='purple' if y > 0 else 'red',
                       edgecolors='black', label='Forecast Round' if t == 1 else "")

    # Decorate plot
    ax.axhline(0, linestyle='--', color='black', alpha=0.5)
    ax.set_xticks(list(hist_times) + list(proj_times) if 'proj_times' in locals() else list(hist_times))
    ax.set_title("📡 Visual Fractal Anchor")
    ax.set_xlabel("Relative Time (Rounds)")
    ax.set_ylabel("MSI Value")
    ax.legend()
    plot_slot = st.empty()
    with plot_slot.container():
        st.pyplot(fig)

    # Echo Signal Summary
    st.metric("🧬 Fractal Match Score", f"{best_score:.3f}")
    if best_future_types:
        st.success(f"📈 Forecasted Round Types: {' '.join(best_future_types)}")
        if best_future_types.count("P") + best_future_types.count("p") >= 2:
            st.info("🔮 Forecast: Surge Mirror Likely")
            st.session_state.last_anchor_type = "Pink"
        elif best_future_types.count("B") >= 2:
            st.warning("⚠️ Blue Collapse Forecast")
            st.session_state.last_anchor_type = "Blue"
        else:
            st.info("🧘 Mixed or Neutral Pattern Incoming")
            st.session_state.last_anchor_type = "Purple"
    else:
        st.session_state.last_anchor_type = "N/A"

# =================== DATA ANALYSIS ========================
@st.cache_data(show_spinner=False)
def analyze_data(data, pink_threshold, window_size):
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= pink_threshold else ("Purple" if x >= 2 else "Blue"))
    df["msi"] = df["score"].rolling(window_size).sum()
    df["momentum"] = df["score"].cumsum()
    
    # Define latest_msi safely
    latest_msi = df["msi"].iloc[-1] if not df["msi"].isna().all() else 0
    latest_tpi = compute_tpi(df, window=window_size)
    
    df["bb_mid"]   = df["msi"].rolling(WINDOW_SIZE).mean()
    df["bb_std"]   = df["msi"].rolling(WINDOW_SIZE).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bandwidth"] = df["bb_upper"] - df["bb_lower"]
    
    # === Detect Squeeze Zones (Low Volatility)
    squeeze_threshold = df["bandwidth"].rolling(10).quantile(0.25)
    df["squeeze_flag"] = df["bandwidth"] < squeeze_threshold
    
    # === Directional Breakout Detector
    df["breakout_up"]   = df["msi"] > df["bb_upper"]
    df["breakout_down"] = df["msi"] < df["bb_lower"]
    
    # === Slope & Acceleration
    df["msi_slope"]  = df["msi"].diff()
    df["msi_accel"]  = df["msi_slope"].diff()

    # MSI CALCULATION (Momentum Score Index)
    window_size = min(window_size, len(df))
    recent_df = df.tail(window_size)
    msi_score = recent_df['score'].mean() if not recent_df.empty else 0
    msi_color = 'green' if msi_score > 0.5 else ('yellow' if msi_score > 0 else 'red')

    # Multi-window BBs on MSI
    df["bb_mid_20"], df["bb_upper_20"], df["bb_lower_20"] = bollinger_bands(df["msi"], 20, 2)
    df["bb_mid_10"], df["bb_upper_10"], df["bb_lower_10"] = bollinger_bands(df["msi"], 10, 1.5)
    df["bb_mid_40"], df["bb_upper_40"], df["bb_lower_40"] = bollinger_bands(df["msi"], 40, 2.5)
    df['bandwidth'] = df["bb_upper_10"] - df["bb_lower_10"]  # Width of the band
    
    # Compute slope (1st derivative) for upper/lower bands
    df['upper_slope'] = df["bb_upper_10"].diff()
    df['lower_slope'] = df["bb_lower_10"].diff()
    
    # Compute acceleration (2nd derivative) for upper/lower bands
    df['upper_accel'] = df['upper_slope'].diff()
    df['lower_accel'] = df['lower_slope'].diff()
    
    # How fast the band is expanding or shrinking
    df['bandwidth_delta'] = df['bandwidth'].diff()
    
    # Pull latest values from the last row
    latest = df.iloc[-1] if not df.empty else pd.Series()
    
    # Prepare and safely round/format outputs, avoiding NoneType formatting
    def safe_round(val, precision=4):
        return round(val, precision) if pd.notnull(val) else None
    
    # Initialize variables
    upper_slope = (0, )
    lower_slope = (0, )
    upper_accel = (0, )
    lower_accel = (0, )
    bandwidth = (0, )
    bandwidth_delta = (0, )
        
    if len(df["score"].fillna(0).values) > 20:
        if upper_slope is not None:
            upper_slope = safe_round(latest.get('upper_slope')) if 'upper_slope' in latest else 0 
            lower_slope = safe_round(latest.get('lower_slope')) if 'lower_slope' in latest else 0
            upper_accel = safe_round(latest.get('upper_accel')) if 'upper_accel' in latest else 0
            lower_accel = safe_round(latest.get('lower_accel')) if 'lower_accel' in latest else 0
            bandwidth = safe_round(latest.get('bandwidth')) if 'bandwidth' in latest else 0
            bandwidth_delta = safe_round(latest.get('bandwidth_delta')) * 100 if 'bandwidth_delta' in latest else 0
                
        
    
    df["bb_squeeze"] = df["bb_upper_10"] - df["bb_lower_10"]
    df["bb_squeeze_flag"] = df["bb_squeeze"] < df["bb_squeeze"].rolling(5).quantile(0.25)
    
    # Harmonic Cycle Estimation
    scores = df["score"].fillna(0).values
    N = len(scores)
    T = 1
    
    # Initialize variables with safe defaults
    dominant_cycle = None
    current_round_position = None
    harmonic_wave = []
    micro_wave = np.zeros(N)
    harmonic_forecast = []
    forecast_times = []
    wave_label = None
    wave_pct = None
    dom_slope = 0
    micro_slope = 0
    eis = 0
    interference = "N/A"
    micro_pct = None
    micro_phase_label = "N/A"
    micro_freq = 0
    dominant_freq = 0
    phase = []
    micro_phase = []
    micro_cycle_len = None
    micro_position = None
    micro_amplitude = 0
    gamma_amplitude = 0
    yf = None
    xf = None
    harmonic_waves = None
    resonance_matrix = None
    resonance_score = None
    tension = None
    entropy = None
    resonance_forecast_vals = None
    
    # Harmonic Analysis
    if N > 0:  # Ensure we have data
        yf = rfft(scores - np.mean(scores))
        xf = rfftfreq(N, T)
        
        # Always detect dominant cycle first
        dominant_cycle = detect_dominant_cycle(scores)
        
        if dominant_cycle:
            current_round_position = len(scores) % dominant_cycle
            wave_label, wave_pct = get_phase_label(current_round_position, dominant_cycle)
            
            # Recompute FFT for wave fitting
            idx_max = np.argmax(np.abs(yf[1:])) + 1 if len(yf) > 1 else 0
            dominant_freq = xf[idx_max] if idx_max < len(xf) else 0
            
            # Harmonic wave fit + forecast
            phase = np.angle(yf[idx_max]) if idx_max < len(yf) else 0
            
            # Harmonic Fit (Past)
            x_past = np.arange(N)  # Safe, aligned x for past
            harmonic_wave = np.sin(2 * np.pi * dominant_freq * x_past + phase)
            dom_slope = np.polyfit(np.arange(N), harmonic_wave, 1)[0] if N > 1 else 0
            
            # Harmonic Forecast (Future)
            forecast_len = 5
            future_x = np.arange(N, N + forecast_len)
            harmonic_forecast = np.sin(2 * np.pi * dominant_freq * future_x + phase)
            forecast_times = [df["timestamp"].iloc[-1] + pd.Timedelta(seconds=5 * i) for i in range(forecast_len)]
            
            # Secondary harmonic (micro-wave) in 8–12 range
            # Micro Wave Detection
            if N > 1 and len(xf) > 1:
                mask_micro = (xf > 0.08) & (xf < 0.15)
                if np.any(mask_micro) and len(np.where(mask_micro)[0]) > 0:
                    micro_indices = np.where(mask_micro)[0]
                    if len(micro_indices) > 0 and len(yf) > max(micro_indices):
                        micro_idx = micro_indices[np.argmax(np.abs(yf[micro_indices]))]
                        micro_freq = xf[micro_idx]
                        micro_phase = np.angle(yf[micro_idx])
                        micro_wave = np.sin(2 * np.pi * micro_freq * np.arange(N) + micro_phase)
                        micro_slope = np.polyfit(np.arange(N), micro_wave, 1)[0] if N > 1 else 0
                        
                        micro_amplitudes = np.abs(yf[micro_indices])
                        micro_amplitude = np.max(micro_amplitudes) if len(micro_amplitudes) > 0 else 0
                        
                        micro_cycle_len = round(1 / micro_freq) if micro_freq else None
                        micro_position = (N - 1) % micro_cycle_len + 1 if micro_cycle_len else None
                        micro_phase_label, micro_pct = get_phase_label(micro_position, micro_cycle_len) if micro_cycle_len else ("N/A", None)
            
            # Energy Integrity Score (EIS)
            blues = len(df[df["score"] < 0])
            purples = len(df[(df["score"] == 1.0) | (df["score"] == 1.5)])
            pinks = len(df[df["score"] >= 2.0])
            eis = (purples * 1 + pinks * 2) - blues
            
            # Alignment test
            if dom_slope > 0 and micro_slope > 0:
                interference = "Constructive (Aligned)"
            elif dom_slope * micro_slope < 0:
                interference = "Destructive (Conflict)"
            else:
                interference = "Neutral or Unclear"
            
            # Channel Bounds (1-STD deviation)
            amplitude = np.std(scores)
            upper_channel = harmonic_forecast + amplitude
            lower_channel = harmonic_forecast - amplitude
            
            gamma_amplitude = np.max(np.abs(yf)) if len(yf) > 0 else 0
    
    # Run resonance analysis if we have enough data
    if N >= 10:  # Need at least 10 rounds
        # Run super-powered harmonic scan
        harmonic_waves, resonance_matrix, resonance_score, tension, entropy = multi_harmonic_resonance_analysis(df)
        
        # Predict next 5 rounds
        resonance_forecast_vals = resonance_forecast(harmonic_waves, resonance_matrix) if harmonic_waves else None
    
    # Return all computed values
    return (df, latest_msi, window_size, recent_df, msi_score, msi_color, latest_tpi, 
            upper_slope, lower_slope, upper_accel, lower_accel, bandwidth, bandwidth_delta, 
            dominant_cycle, current_round_position, wave_label, wave_pct, dom_slope, micro_slope, 
            eis, interference, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, 
            micro_pct, micro_phase_label, micro_freq, dominant_freq, phase, gamma_amplitude, 
            micro_amplitude, micro_phase, micro_cycle_len, micro_position, harmonic_waves, 
            resonance_matrix, resonance_score, tension, entropy, resonance_forecast_vals)

# =================== MSI CHART PLOTTING ========================
def plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times):
    if len(df) < 2:
        st.warning("Need at least 2 rounds to plot MSI chart.")
        return
        
    # MSI with Bollinger Bands
    st.subheader("MSI with Bollinger Bands")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["msi"], label="MSI", color='black')
    
    # BB lines
    ax.plot(df["timestamp"], df["bb_upper"], linestyle='--', color='green')
    ax.plot(df["timestamp"], df["bb_lower"], linestyle='--', color='red')
    ax.fill_between(df["timestamp"], df["bb_lower"], df["bb_upper"], color='gray', alpha=0.1)
    
    # Highlight squeeze
    ax.scatter(df[df["squeeze_flag"]]["timestamp"], df[df["squeeze_flag"]]["msi"], color='purple', label="Squeeze", s=20)
    
    # Highlight breakouts
    ax.scatter(df[df["breakout_up"]]["timestamp"], df[df["breakout_up"]]["msi"], color='lime', label="Breakout ↑", s=20)
    ax.scatter(df[df["breakout_down"]]["timestamp"], df[df["breakout_down"]]["msi"], color='red', label="Breakout ↓", s=20)
    
    ax.set_title("📊 MSI Volatility Tracker")
    ax.legend()
    st.pyplot(fig)

# =================== MAIN APP FUNCTIONALITY ========================
# =================== FLOATING ADD ROUND UI ========================


# Fast entry mode UI - simplified UI for mobile/quick decisions
def fast_entry_mode_ui(pink_threshold):
    st.markdown("### ⚡ FAST ENTRY MODE")
    st.markdown("Quick enter rounds for rapid decision making")
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("➕ Blue (1.5x)", use_container_width=True):
            st.session_state.roundsc.append({
                "timestamp": datetime.now(),
                "multiplier": 1.5,
                "score": -1
            })
            st.rerun()
    
    with cols[1]:
        if st.button("➕ Purple (2x)", use_container_width=True):
            st.session_state.roundsc.append({
                "timestamp": datetime.now(),
                "multiplier": 2.0,
                "score": 1
            })
            st.rerun()
    
    with cols[2]:
        if st.button(f"➕ Pink ({pink_threshold}x)", use_container_width=True):
            st.session_state.roundsc.append({
                "timestamp": datetime.now(),
                "multiplier": pink_threshold,
                "score": 2
            })
            st.rerun()

# =================== MAIN APP CODE ========================
# Round Entry form
col_entry, col_hud = st.columns([2, 1])
with col_entry:
    st.subheader("📊 Manual Round Entry")
    mult = st.number_input("Enter round multiplier", min_value=0.01, step=0.01, value=st.session_state.current_mult)
    st.session_state.current_mult = mult
    
    score_value = 2 if mult >= PINK_THRESHOLD else (1 if mult >= 2.0 else -1)
    round_type = "🔴 Pink" if mult >= PINK_THRESHOLD else ("🟣 Purple" if mult >= 2.0 else "🔵 Blue")
    
    st.info(f"This will be recorded as a {round_type} round with score {score_value}")
    
    if st.button("➕ Add Round", use_container_width=True):
        st.session_state.roundsc.append({
            "timestamp": datetime.now(),
            "multiplier": mult,
            "score": score_value
        })
        st.rerun()

# Display fast entry mode if enabled
if FAST_ENTRY_MODE:
    fast_entry_mode_ui(PINK_THRESHOLD)

# Convert rounds to DataFrame
df = pd.DataFrame(st.session_state.roundsc)

# Main App Logic
if not df.empty:
    # Run analysis
    (df, latest_msi, window_size, recent_df, msi_score, msi_color, latest_tpi, 
     upper_slope, lower_slope, upper_accel, lower_accel, bandwidth, bandwidth_delta, 
     dominant_cycle, current_round_position, wave_label, wave_pct, dom_slope, micro_slope, 
     eis, interference, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, 
     micro_pct, micro_phase_label, micro_freq, dominant_freq, phase, gamma_amplitude, 
     micro_amplitude, micro_phase, micro_cycle_len, micro_position, harmonic_waves, 
     resonance_matrix, resonance_score, tension, entropy, resonance_forecast_vals) = analyze_data(df, PINK_THRESHOLD, WINDOW_SIZE)
    
    # Check if we completed a cycle
    if dominant_cycle and current_round_position == 0 and 'last_position' in st.session_state:
        if st.session_state.last_position > 0:  # We completed a cycle
            st.session_state.completed_cycles += 1
    st.session_state.last_position = current_round_position if current_round_position is not None else 0
    
    # Calculate RRQI
    rrqi_val = rrqi(df, 30)
    
    # Display metrics
    with col_hud:
        st.metric("Rounds Recorded", len(df))
        
    
    # Plot MSI Chart
    plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times)
    
    # === SHOW MORLET PANEL FOR LARGER DATASETS ===
    if len(df) >= 20:
        with st.expander("🌀 Quantum String Resonance Analyzer", expanded=False):
            
            st.markdown("## 🌊 Morlet Wavelet Burst Tracker")
            scores = df["score"].fillna(0).values
            coeffs, power, freqs, scales = morlet_wavelet_power(scores)
        
            fig, ax = plt.subplots(figsize=(12, 6))
            t = np.arange(power.shape[1])  # Time axis
        
            im = ax.imshow(
                power, extent=[t[0], t[-1], scales[-1], scales[0]],
                aspect='auto', cmap='plasma'
            )
            ax.set_title("Morlet Power Map (Time × Scale)")
            ax.set_ylabel("Wavelet Scale (lower = faster)")
            ax.set_xlabel("Round Index")
            fig.colorbar(im, ax=ax, label="Power")
        
            st.pyplot(fig)
        
            # Optional: Phase tracking signal
            mean_energy = np.mean(power, axis=0)
            st.line_chart(mean_energy, height=200)
            st.caption("Average Burst Energy Across Scales (watch for peaks)")

        

        st.markdown("### 🧬 Fractal Nonlinear Resonance Engine (FNR)")

        fnr_metrics = compute_fnr_index_from_morlet(power, scales)
        
        if fnr_metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("🔀 FNR Index", fnr_metrics["FNR_index"])
            col2.metric("📐 Cosine Phase", fnr_metrics["Phase_cosine_similarity"])
            col3.metric("🧭 Alignment Type", fnr_metrics["alignment"])
        
            if fnr_metrics["alignment"] == "Constructive":
                st.success("💥 Constructive Interference — True Surge Field Detected")
            elif fnr_metrics["alignment"] == "Destructive":
                st.error("🌪 Destructive Phase — Collapse Pressure Likely")
            else:
                st.info("🧘 Neutral Field — Moderate Risk Zone")
        else:
            st.warning("Not enough wavelet data to compute FNR")

        

    # === QUANTUM STRING DASHBOARD ===
    with st.expander("🌀 Quantum String Resonance Analyzer", expanded=False):
        st.subheader("🧵 Multi-Harmonic Resonance Matrix")
        
        if resonance_matrix is not None:
            # Colorful resonance grid
            fig, ax = plt.subplots()
            cax = ax.matshow(resonance_matrix, cmap='viridis')
            fig.colorbar(cax, label='Resonance Strength')
            ax.set_xticks(range(len(resonance_matrix)))
            ax.set_yticks(range(len(resonance_matrix)))
            ax.set_xticklabels([f'H{i+1}' for i in range(len(resonance_matrix))])
            ax.set_yticklabels([f'H{i+1}' for i in range(len(resonance_matrix))])
            st.pyplot(fig)
            
            # Show quantum metrics
            string_metrics_panel(tension, entropy, resonance_score)
            
            # Forecast chart
            st.subheader("🔮 Resonance Forecast")
            if resonance_forecast_vals is not None:
                st.line_chart(pd.DataFrame({
                    'Forecast': resonance_forecast_vals,
                    'Confidence': [x * 0.7 for x in resonance_forecast_vals]
                }))
    
    # === SHOW THRE PANEL IF ENABLED ===
    if show_thre: 
        with st.expander("🔬 True Harmonic Resonance Engine (THRE)", expanded=False):
            (df, latest_rds, latest_delta) = thre_panel(df)
            # Display fast entry mode if enabled
            
    # === LIVE PROBABILITY PANEL ===
            if len(df) >= 20:
                st.markdown("### 🎯 Surge Probability Engine (THRE + FNR Fusion)")
                if fnr_metrics and latest_rds is not None and latest_delta is not None :
                    
                     
                    surge_prob, components = compute_surge_probability(
                    thre_val=latest_rds,
                    delta_slope=latest_delta,
                    fnr_index=fnr_metrics["FNR_index"]
                    )
                    surge_score = surge_prob  # now it's just a number like 0.84

                    col1, col2 = st.columns([1, 2])
                    col1.metric("🔮 Surge Probability", f"{int(surge_score * 100)}%")

                    col2.progress(surge_prob)
        
                    st.markdown("Component Breakdown")
                    st.write(f"**THRE Signal**: {components['thre_component']}")
                    st.write(f"**FNR Alignment**: {components['fnr_component']}")
                    st.write(f"**THRE Δ Slope**: {components['slope_component']}")
        
                     # Optional guidance output
                    if surge_prob >= 0.8:
                        st.success("💖 Pink Entry Confirmed — Surge Stack is Aligned")
                    elif surge_prob >= 0.6:
                        st.info("🟣 Purple Entry Likely — Some Constructive Field Detected")
                    elif surge_prob <= 0.3:
                        st.warning("🔵 Risk of Collapse — Weak Field Detected")
                    else:
                        st.info("⚪ Neutral Field — Entry Requires Caution")
            
            
       
        
             
    
    # === SHOW COSINE PHASE PANEL IF ENABLED ===
    if show_cos_panel: 
        with st.expander("🌀 Cosine Phase Alignment Panel", expanded=False):
            cos_phase_panel(df, dominant_freq, micro_freq, phase, micro_phase)
    
    # === SHOW HARMONIC ROUND PREDICTOR ===
    if len(df) >= 20:
        with st.expander("🔮 Harmonic Round Predictor", expanded=False):
            classification, action, energy_index = classify_next_round(
                resonance_forecast_vals, tension, entropy, resonance_score
            )
            st.metric("Next Round Prediction", classification)
            st.metric("Suggested Action", action)
            st.metric("Resonance Energy Index", round(energy_index, 4))
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("🎯 Coherence", f"{resonance_score:.4f}")
            with col2: st.metric("🎸 Tension", f"{tension:.4f}")
            with col3: st.metric("📊 Entropy", f"{entropy:.4f}")
    
    # === SHOW RQCF PANEL IF ENABLED ===
    if show_rqcf and not FAST_ENTRY_MODE:
        with st.expander("🔮 RQCF Panel: Recursive Quantum Chain Forecast", expanded=False):
            chains = run_rqcf(df["score"].fillna(0).values)
            for chain in chains:
                st.markdown(f"**{chain['branch']}**")
                for i, (val, label) in enumerate(chain["forecast"]):
                    st.markdown(f"- Step {i+1}: `{label}` → `{val}`")
    
    # === SHOW FPM PANEL IF ENABLED ===
    if show_fpm: 
        with st.expander("🧬 Fractal Pulse Matcher Panel (FPM)", expanded=False):
            fpm_panel(df)
    
    # === SHOW FRACTAL ANCHOR IF ENABLED ===
    if show_anchor: 
        with st.expander("🔗 Fractal Anchoring Visualizer", expanded=False):
            fractal_anchor_visualizer(df)
    
    # === DECISION HUD PANEL ===
    decision_hud_panel(
        dominant_phase=wave_label or "N/A",
        dominant_pct=wave_pct or 0,
        micro_phase=micro_phase_label or "N/A",
        micro_pct=micro_pct or 0,
        resonance_score=resonance_score if 'resonance_score' in locals() else None,
        fractal_match_type=st.session_state.get("last_fractal_match"),
        anchor_forecast_type=st.session_state.get("last_anchor_type")
    )
    
    # === RRQI STATUS ===
    st.metric("🧠 RRQI", rrqi_val, delta="Last 30 rounds")
    if rrqi_val >= 0.3:
        st.success("🔥 Happy Hour Detected — Tactical Entry Zone")
    elif rrqi_val <= -0.2:
        st.error("⚠️ Dead Zone — Avoid Aggressive Entries")
    else:
        st.info("⚖️ Mixed Zone — Scout Cautiously")
    
    # === WAVE ANALYSIS PANEL ===
    with st.expander("🔊 Wave Analysis", expanded=False):
        st.subheader("📡 Harmonic Phase Tracker")
        if wave_label is not None and wave_pct is not None:
            st.metric("Dominant Cycle Length", f"{dominant_cycle} rounds")
            st.metric("Wave Position", f"Round {current_round_position} of {dominant_cycle}")
            st.metric("Wave Phase", f"{wave_label} ({wave_pct:.1f}%)")
            st.metric("EIS", eis)
            st.metric("Dominant Slope", f"{dom_slope:.3f}")
            st.metric("Micro Slope", f"{micro_slope:.3f}")
            st.metric("Completed Cycles", st.session_state.completed_cycles)
            st.info(f"ℹ️ Wave Interference: {interference}")
        else:
            st.metric("Wave Phase", "N/A")
    
        st.subheader("📉 Micro Harmonic Phase Tracker")
        if micro_phase_label != "N/A":
            st.metric("Micro Cycle Length", f"{micro_cycle_len} rounds")
            st.metric("Micro Wave Position", f"Round {micro_position} of {micro_cycle_len}")
            st.metric("Micro Wave Phase", f"{micro_phase_label} ({micro_pct:.1f}%)")
        else:
            st.info("Micro Wave Phase: N/A — Not enough data")
    
        if micro_amplitude > 0:
            st.metric("Micro Frequency", f"{micro_freq:.4f}")
            st.metric("Micro Amplitude", f"{micro_amplitude:.4f}")
            st.progress(min(1.0, micro_amplitude / gamma_amplitude) if gamma_amplitude > 0 else 0)
        else:
            st.warning("Micro wave not detected in current data")
    
    # === BOLLINGER BANDS STATS ===
    with st.expander("💹 Bollinger Bands Stats", expanded=False):
        st.subheader("💹 Bollinger Bands Stats")
        if upper_slope is not None:
            st.metric("Upper Slope", f"{upper_slope[0]}%")
            st.metric("Upper Acceleration", f"{upper_accel[0]}%")
            st.metric("Lower Slope", f"{lower_slope[0]}%")
            st.metric("Lower Acceleration", f"{lower_accel[0]}%")
            st.metric("Bandwidth", f"{bandwidth[0]} Scale (0-20)")
            st.metric("Bandwidth Delta", f"{bandwidth_delta[0]}% shift from last round")
        else:
            st.info("Not enough data for Bollinger Band metrics")
    
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
    
    # === LOG/EDIT ROUNDS ===
    with st.expander("📄 Review / Edit Recent Rounds", expanded=False):
        edited = st.data_editor(df.tail(30), use_container_width=True, num_rows="dynamic")
        if st.button("✅ Commit Edits"):
            st.session_state.roundsc = edited.to_dict('records')
            st.rerun()

else:
    st.info("Enter at least 1 round to begin analysis.")


