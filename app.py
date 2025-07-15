import streamlit as st
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import sklearn
#import pywt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from datetime import timedelta
#import collections
from collections import defaultdict
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert
from scipy.signal import savgol_filter
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from matplotlib import gridspec
#from thre_fused_tdi_module import plot_thre_fused_tdi
#import morlet_phase_enhancement
#from morlet_phase_enhancement import morlet_phase_panel

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

if "alignment_score_history" not in st.session_state:
    st.session_state["alignment_score_history"] = []

# Initialize session state for predictions
if 'qfe_predictions' not in st.session_state:
    st.session_state.qfe_predictions = {}
    
if 'qfe_accuracy' not in st.session_state:
    st.session_state.qfe_accuracy = {
        'total': 0,
        'correct': 0,
        'last_checked': -1
    }


# ================ CONFIGURATION SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ QUANTUM PARAMETERS")
    WINDOW_SIZE = st.slider("MSI Window Size", 5, 100, 20)
    PINK_THRESHOLD = st.number_input("Pink Threshold", value=10.0)
    STRICT_RTT = st.checkbox("Strict RTT Mode", value=False)
    selected_window = st.sidebar.selectbox(
    "Select Fibonacci spirals Window",
    options=[3, 5, 8, 13, 21, 34, 55],
    index=5  # default to 34
    )
    fib_window = st.sidebar.selectbox(
    "Fibonacci Envelope Window Size",
    options=[5, 8, 13, 21, 34],
    index=2  # default to 13
    )

    st.sidebar.subheader("FLP Projection Layers")
    fib_layer_options = [3, 5, 8, 13, 21, 34, 55]
    selected_fib_layers = st.sidebar.multiselect(
        "Choose Fibonacci Gaps (in rounds) for Projection",
        options=fib_layer_options,
        default=[5, 8, 13, 21, 34]
    )

    st.sidebar.subheader("MSI Multi-Window Settings")

    default_fib_windows = [3, 5, 8, 13, 21, 34]
    
    selected_msi_windows = st.sidebar.multiselect(
        "Select MSI Window Sizes (Fibonacci)",
        options=default_fib_windows,
        default=[13, 21]
    )
    st.sidebar.subheader("🧭 Set Weights for Each MSI Window")

    window_weights = []
    for window in selected_msi_windows:
        weight = st.sidebar.slider(
            f"Weight for MSI {window}",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
        window_weights.append(weight)

    # ---------------------------------
    # Sidebar Toggles for Fib Retracement
    # ---------------------------------
    st.sidebar.subheader("📐 Fibonacci Retracement Settings")
    fib_msi_window = st.sidebar.selectbox(
    "MSI Window for Fib Retracement",
    options=[3, 5, 8, 13, 21, 34, 55],
    index=2
    )

    fib_lookback_window = st.sidebar.selectbox(
    "Lookback Window (Rounds)",
    options=[3, 5, 8, 13, 21, 34, 55],
    index=3
    )
    
    show_fib_retracement = st.sidebar.checkbox(
        "📏 Show Fib Retracements", value=True
    )
    show_fib_ext = st.sidebar.checkbox(
        "📏 Show Fib extentsions", value=True
    )

    st.sidebar.subheader("📐 Multi-Window Fibonacci Analysis")
    multi_fib_windows = st.sidebar.multiselect(
        "Select Fib Lookback Windows",
        options=[5, 8, 13, 21, 34, 55],
        default=[13, 21, 34]
    )
    show_multi_fib_analysis = st.sidebar.checkbox(
        "📊 Show Multi-Window Fib Analysis",
        value=True
    )

    st.sidebar.subheader("🎯 Fib Alignment Score Engine Settings")

    fib_alignment_window = st.sidebar.selectbox(
        "Lookback Window (Rounds)",
        options=[21, 34, 55],
        index=1
    )
    
    fib_alignment_tolerance = st.sidebar.slider(
        "Gap Tolerance",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
    st.header("📉 Indicator Visibility")

    show_supertrend = st.checkbox("🟢 Show SuperTrend", value=True)
    show_ichimoku   = st.checkbox("☁️ Show Ichimoku (Tenkan/Kijun)", value=True)
    show_fibo   = st.checkbox("💫 Show FIB modules", value=True)
    show_bb   = st.checkbox("🌈 Show BB bands", value=True)
    show_fibo_bands   = st.checkbox("📏 Show FIB  bands", value=True)
    show_msi_res = st.checkbox("💹 MSI Res", value=True)

    st.header("📊 PANEL TOGGLES")
    FAST_ENTRY_MODE = st.checkbox("⚡ Fast Entry Mode", value=False)
    
    if st.button("🔄 Full Reset", help="Clear all historical data"):
        st.session_state.roundsc = []
        st.rerun()
        
    # Clear cached functions
    if st.button("🧹 Clear Cache", help="Force harmonic + MSI recalculation"):
        st.cache_data.clear()  # Streamlit's built-in cache clearer
        st.success("Cache cleared — recalculations will run fresh.")
        
    RANGE_WINDOW = st.sidebar.selectbox(
        "Range Lookback Window (Rounds)",
        options=[3, 5, 8, 13, 21, 34, 55],
        index=3
        )
    VOLATILITY_THRESHOLDS = {
    'micro': 1.5,
    'meso': 3.0,
    'macro': 5.0
    }
    VOLATILITY_THRESHOLDS['micro'] = st.number_input("Micro Threshold", value=1.5)
    VOLATILITY_THRESHOLDS['meso'] = st.number_input("Meso Threshold", value=3.0)
    VOLATILITY_THRESHOLDS['macro'] = st.number_input("Macro Threshold", value=5.0)
     
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


    
FIB_NUMBERS = [3, 5, 8, 13, 21, 34, 55]
FIBONACCI_NUMBERS = [3, 5, 8, 13, 21, 34, 55]
ENVELOPE_MULTS = [1.0, 1.618, 2.618]
FIB_GAPS = [3, 5, 8, 13, 21, 34, 55]

class NaturalFibonacciSpiralDetector:
    def __init__(self, df, window_size=34):
        self.df = df.tail(window_size).reset_index(drop=True)
        self.df["round_index"] = self.df.index
        self.window_size = window_size
        self.spiral_candidates = []

    def detect_spirals(self):
        score_types = [-1, 1, 2]  # Blue, Purple, Pink
        scores = self.df["score"].fillna(0).values

        for score_type in score_types:
            idxs = [i for i, val in enumerate(scores) if val == score_type]

            # Check for Fibonacci Gaps within this score type
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    gap = idxs[j] - idxs[i]
                    if gap in FIB_NUMBERS:
                        self.spiral_candidates.append({
                            "score_type": score_type,
                            "start_idx": idxs[i],
                            "end_idx": idxs[j],
                            "gap": gap,
                            "center_idx": (idxs[i] + idxs[j]) // 2
                        })

        return self._select_strongest_spirals()

    def _select_strongest_spirals(self, max_spirals=3):
        centers = defaultdict(int)

        # Count how often each center index appears
        for candidate in self.spiral_candidates:
            centers[candidate["center_idx"]] += 1

        # Sort by frequency (descending)
        top_centers = sorted(centers.items(), key=lambda x: -x[1])[:max_spirals]
        spiral_centers = []

        for center_idx, freq in top_centers:
            ts = self.df.loc[center_idx, "timestamp"]
            label = self._label_score(self.df.loc[center_idx, "score"])
            spiral_centers.append({
                "center_index": center_idx,
                "round_index": center_idx,
                "timestamp": ts,
                "score_type": self.df.loc[center_idx, "score"],
                "label": label,
                "confirmations": freq
            })

        return spiral_centers

    def _label_score(self, s):
        return { -1: "Blue", 1: "Purple", 2: "Pink" }.get(s, "Unknown")
    
def get_spiral_echoes(spiral_centers, df, gaps=[3, 5, 8, 13]):
    echoes = []

    for sc in spiral_centers:
        base_idx = sc["round_index"]

        for gap in gaps:
            echo_idx = base_idx + gap
            if echo_idx < len(df):
                echoes.append({
                    "echo_round": echo_idx,
                    "source_round": base_idx,
                    "timestamp": df.loc[echo_idx, "timestamp"],
                    "source_label": sc["label"],
                    "gap": gap
                })

    return echoes    

def project_true_forward_flp(spiral_centers, fib_layers=[6, 12, 18, 24, 30], max_rounds=None):
    """
    Create a forward projection map from spiral centers using Fibonacci gaps.
    Returns a list of watchlist targets.
    """

    watchlist = []

    for sc in spiral_centers:
        base_idx = sc["round_index"]

        for gap in fib_layers:
            target_idx = base_idx + gap
            if max_rounds is None or target_idx < max_rounds:
                watchlist.append({
                    "source_round": base_idx,
                    "source_label": sc["label"],
                    "gap": gap,
                    "target_round": target_idx
                })

    return watchlist

def compute_resonance(row, selected_windows, weights=None):
    if weights is None:
        weights = [1] * len(selected_windows)
    weighted_sum = 0
    weight_total = sum(weights)
    
    for i, window in enumerate(selected_windows):
        sign_col = f"sign_{window}"
        weighted_sum += row[sign_col] * weights[i]
    
    resonance_score = weighted_sum / weight_total
    return round(resonance_score, 2) 

def compute_slope_agreement(slopes):
    """Percentage of slopes in the same direction."""
    signs = [np.sign(s) for s in slopes]
    pos_count = signs.count(1)
    neg_count = signs.count(-1)
    return round(max(pos_count, neg_count) / len(signs), 2)

def compute_ordering_score(msi_values):
    """Measures hierarchy — are shorter windows leading longer?"""
    mismatches = sum(1 for i in range(len(msi_values)-1) if msi_values[i] < msi_values[i+1])
    return round(1 - (mismatches / (len(msi_values) - 1)), 2)

def detect_slope_inflections(slope_series):
    """Counts number of sign flips in slopes."""
    inflections = 0
    for i in range(1, len(slope_series)):
        if np.sign(slope_series[i]) != np.sign(slope_series[i-1]):
            inflections += 1
    return inflections

def compute_weighted_energy(msi_values, window_sizes):
    weights = []
    for w in window_sizes:
        if w <= 8:
            weights.append(0.5)
        elif w <= 21:
            weights.append(1.0)
        else:
            weights.append(1.5)
    weighted = [v * w for v, w in zip(msi_values, weights)]
    return sum(weighted) / sum(weights)

def dynamic_feb_bands(ma_value, phase_score):
    """Phase-weighted dynamic envelope bands."""
    if phase_score >= 0.75:
        mults = ENVELOPE_MULTS
    elif phase_score >= 0.5:
        mults = [0.85, 1.3, 2.0]
    else:
        mults = [0.7, 1.0, 1.5]
    return [round(ma_value * m, 3) for m in mults]

def check_envelope_breakouts(msi_value, bands):
    """Flags for each breakout level."""
    return {
        "breakout_1": msi_value >= bands[0],
        "breakout_1_618": msi_value >= bands[1],
        "breakout_2_618": msi_value >= bands[2],
    }

def compute_volatility(series, window=5):
    """Std deviation over recent rounds."""
    return round(series[-window:].std(), 3)

def compute_gap_since_last_pink(current_index, last_pink_index):
    """How many rounds since last surge/pink?"""
    if last_pink_index is None:
        return None
    return max(1, current_index - last_pink_index)

def fib_gap_alignment(gap):
    """How closely does gap match Fibonacci numbers?"""
    if gap is None:
        return 0.5
    closest = min(FIBONACCI_NUMBERS, key=lambda x: abs(x - gap))
    alignment = 1 - (abs(closest - gap) / max(FIBONACCI_NUMBERS))
    return round(max(0, alignment), 2)

def map_phase_label(score):
    if score >= 0.75:
        return "Ascent"
    elif score >= 0.5:
        return "Birth"
    elif 0.4 <= score < 0.5:
        return "Peak"
    elif 0.25 <= score < 0.4:
        return "Post-Peak"
    else:
        return "Collapse"

def compute_custom_phase_score(
    current_round_index,
    last_pink_index,
    msi_values,
    slopes,
    window_sizes
):
    # Ordering Score
    ordering_score = compute_ordering_score(msi_values)
    
    # Slope Agreement
    slope_alignment = compute_slope_agreement(current_slopes)
    
    # Weighted Energy
    weighted_energy = compute_weighted_energy(msi_values, window_sizes)

    gap = compute_gap_since_last_pink(current_round_index, last_pink_index)
    fib_alignment = fib_gap_alignment(gap)

     # Final Phase Score (tunable weights)
    phase_score = (
        ordering_score * 0.3 +
        slope_alignment * 0.2 +
        weighted_energy * 0.3 +
        fib_alignment * 0.2
    )
    phase_score = round(min(max(phase_score, 0.0), 1.0), 3)

    phase_label = map_phase_label(phase_score)

    return {
        'ordering_score': ordering_score,
        'slope_alignment': slope_alignment,
        'weighted_energy': weighted_energy,
        "gap_since_last_pink": gap,
        "fib_gap_alignment": fib_alignment,
        'phase_score': phase_score,
        'phase_label': phase_label
    }

def estimate_regime_length_class(gap):
    """Classifies regime by gap size."""
    if gap is None:
        return "Unknown", None
    if gap <= 8:
        return "Micro", 8
    elif gap <= 21:
        return "Meso", 21
    else:
        return "Macro", 34

def compute_spiral_projection_windows(current_round, fib_numbers=FIBONACCI_NUMBERS):
    """Predict future likely shift windows."""
    return [current_round + f for f in fib_numbers]

def detect_trap_cluster_state(recent_gaps, fib_numbers=FIBONACCI_NUMBERS):
    """
    Determines if recent pink gaps are clustering tightly in Fibonacci intervals,
    which suggests bait/trap regimes designed to harvest bets.
    """
    if len(recent_gaps) < 3:
        return "Unknown"

    # Count how many gaps are in small Fib range
    tight_cluster_count = sum(1 for g in recent_gaps if g in [3,5,8])
    if tight_cluster_count >= 2:
        return "High Trap Probability"

    # Check for expansion pattern
    expansion_gaps = [g for g in recent_gaps if g > 13]
    if len(expansion_gaps) >= 2:
        return "Expansion Zone (Dry Trap)"

    return "Neutral"


def classify_phase_label(slope_agreement, ordering_score, inflections, phase_score, volatility):
    """
    Label the current phase of the regime based on MSI slope structure, inflections,
    and volatility.
    """
    if slope_agreement >= 0.8 and ordering_score >= 0.75 and inflections <= 1 and phase_score >= 0.75:
        return "Ascent"

    if slope_agreement >= 0.5 and ordering_score >= 0.5 and inflections <= 2 and phase_score >= 0.5:
        return "Birth"

    if inflections >= 3 or volatility > 1.0:
        return "Peak / Range"

    if slope_agreement <= 0.2 and ordering_score <= 0.2:
        return "Collapse / Blue Train"

    return "Unclassified"

def classify_regime_state(
    current_round_index,
    last_pink_index,
    recent_scores,
    current_msi_values,
    current_slopes,
    slope_history_series,
    phase_score,
    recent_gap_history
):
    """
    Complete regime classification for current round.
    """

    # Core Slope Features
    slope_agreement = compute_slope_agreement(current_slopes)
    ordering_score = compute_ordering_score(current_msi_values)
    inflections = sum([detect_slope_inflections(s) for s in slope_history_series])
    volatility = compute_volatility(recent_scores)

    # Envelope Breakout Detection
    ma_value = np.mean(recent_scores)
    bands = dynamic_feb_bands(ma_value, phase_score)
    msi_mean_value = np.mean(current_msi_values)
    envelope_flags = check_envelope_breakouts(msi_mean_value, bands)

    # Gap and Fibonacci Alignment
    gap = compute_gap_since_last_pink(current_round_index, last_pink_index)
    gap_alignment = fib_gap_alignment(gap)
    regime_type, estimated_length = estimate_regime_length_class(gap)
    current_pos_in_regime = gap if gap else 0
    rounds_to_shift = estimated_length - current_pos_in_regime if estimated_length else None

    spiral_forecast = compute_spiral_projection_windows(current_round_index)
    trap_cluster_state = detect_trap_cluster_state(recent_gap_history)

    # Phase Classification
    phase_label = classify_phase_label(slope_agreement, ordering_score, inflections, phase_score, volatility)

    # Final Result
    result = {
        "regime_type": regime_type,
        "estimated_length": estimated_length,
        "phase_label": phase_label,
        "phase_score": phase_score,
        "current_round_in_regime": current_pos_in_regime,
        "rounds_to_next_shift": rounds_to_shift,
        "gap_since_last_pink": gap,
        "fib_gap_alignment": gap_alignment,
        "spiral_projection_windows": spiral_forecast,
        "trap_cluster_state": trap_cluster_state,
        "slope_agreement": slope_agreement,
        "ordering_score": ordering_score,
        "slope_inflections": inflections,
        "volatility": volatility,
        "envelope_flags": envelope_flags
    }

    return result

# ============================
# MSI FIBONACCI RETRACEMENT MODULE
# ============================

def calculate_fibonacci_retracements(msi_series, fib_lookback_window):
    """
    Calculate Fibonacci retracement and extension levels
    from swing high/low over user-specified lookback window.
    """
    recent = msi_series.tail(fib_lookback_window).dropna()
    if recent.empty or len(recent) < 2:
        return None

    swing_high = recent.max()
    swing_low = recent.min()

    if swing_high == swing_low:
        # Avoid division by zero when flat
        return None

    # Compute standard retracement levels
    retracements = {
        "0.0": round(swing_low, 3),
        "0.236": round(swing_high - 0.236 * (swing_high - swing_low), 3),
        "0.382": round(swing_high - 0.382 * (swing_high - swing_low), 3),
        "0.5": round((swing_high + swing_low) / 2, 3),
        "0.618": round(swing_high - 0.618 * (swing_high - swing_low), 3),
        "0.786": round(swing_high - 0.786 * (swing_high - swing_low), 3),
        "1.0": round(swing_high, 3)
    }

    # Compute extension levels
    range_ = swing_high - swing_low
    extensions = {
        "1.618": round(swing_high + 0.618 * range_, 3),
        "2.618": round(swing_high + 1.618 * range_, 3),
        "3.618": round(swing_high + 2.618 * range_, 3),
        "-0.618": round(swing_low - 0.618 * range_, 3),
        "-1.618": round(swing_low - 1.618 * range_, 3),
        "-2.618": round(swing_low - 2.618 * range_, 3)
    }

    return retracements, extensions, swing_high, swing_low

def compute_multi_window_fib_retracements(df, msi_column, windows):
    """
    For each selected lookback window, compute retracement levels.
    Returns a dict of window -> retracement levels.
    """
    results = {}
    for w in windows:
        res = calculate_fibonacci_retracements(df[msi_column], w)
        if res:
            retrace, ext, high, low = res
            results[w] = {
                "retracements": retrace,
                "extensions": ext,
                "swing_high": high,
                "swing_low": low
            }
    return results

def compute_fib_alignment_score(df, fib_threshold=10.0, lookback_window=34, tolerance=1.0):
    """
    Compute a score 0–1 for how well the sequence of recent 'pink' rounds
    aligns to Fibonacci intervals.
    #- fib_threshold: multiplier value considered a pink.
    #- lookback_window: number of rounds to analyze.
    #- tolerance: +/- range allowed when matching Fib gaps.
    """
    if len(df) < lookback_window:
        return None, []

    recent_df = df.tail(lookback_window).copy()
    recent_df = recent_df.reset_index(drop=True)
    recent_df['relative_index'] = recent_df.index

    # Find pink rounds in recent window
    pink_indexes = recent_df[recent_df['multiplier'] >= fib_threshold]['relative_index'].tolist()

    if len(pink_indexes) < 2:
        return 0.0, []

    # Compute gaps between pink rounds
    gaps = [pink_indexes[i+1] - pink_indexes[i] for i in range(len(pink_indexes)-1)]

    # For each gap, score it by closeness to Fib sequence
    scores = []
    for gap in gaps:
        diffs = [abs(gap - fib) for fib in FIB_GAPS]
        min_diff = min(diffs)
        if min_diff <= tolerance:
            # Perfect or near-perfect match
            scores.append(1.0)
        else:
            # Score decays with distance
            decay = np.exp(-min_diff / tolerance)
            scores.append(decay)

    # Average score
    if scores:
        alignment_score = np.clip(np.mean(scores), 0, 1)
    else:
        alignment_score = 0.0

    return round(alignment_score, 3), gaps

class QuantumFibonacciEntanglement:
    def __init__(self, multiplier_sequence: list):
        self.multipliers = multiplier_sequence
        self.fib_windows = [3, 5, 8, 13, 21]
        self.golden_ratio = 1.6180339887
        self.pressure_history = []

    def _calculate_pressure_wavefront(self, window):
        if len(self.multipliers) < window:
            return 0, 0, 0, 0, 0

        segment = self.multipliers[-window:]
        surge_force, resistance, momentum = 0, 0, 0

        # Surge Detection with Temporal Decay
        decay_weights = np.linspace(1.5, 0.5, len(segment))
        for i, m in enumerate(segment):
            if m >= 10:
                surge_force += 3.5 * decay_weights[i]
            elif m >= 2:
                surge_force += 1.2 * decay_weights[i]

        # Blue Trap Resistance
        for i, m in enumerate(segment):
            if m < 2:
                fib_weight = self.golden_ratio ** (i / len(segment))
                resistance += (2 - m) * fib_weight

        # Momentum Tracking with Phase Acceleration
        slopes = []
        for i in range(1, len(segment)):
            delta = segment[i] - segment[i - 1]
            slopes.append(delta)
            if delta > 0:
                momentum += 1.5 if segment[i] > 2 else 0.8
            else:
                momentum -= 0.6

        acceleration = np.diff(slopes).mean() if len(slopes) > 2 else 0

        # Fib Distortion Energy
        expected_ratio = self.golden_ratio
        local_max = max(segment)
        local_min = min(segment)
        actual_ratio = (local_max + 0.01) / (local_min + 0.01)
        fib_distortion = abs(expected_ratio - actual_ratio) * 0.5

        return surge_force, resistance, momentum, acceleration, fib_distortion

    def adaptive_fpi_dashboard(self):
        wavefront_data = []

        for i, win in enumerate(self.fib_windows):
            surge, resistance, momentum, acceleration, distortion = self._calculate_pressure_wavefront(win)
            volatility = np.std(self.multipliers[-win:]) if len(self.multipliers) >= win else 0

            # Composite Surge Potential
            surge_potential = (
                surge
                - 0.6 * resistance
                + 0.9 * momentum
                + 1.2 * acceleration
                - 0.8 * distortion
                + 0.4 * volatility
            )

            adaptive_threshold = 0.6 + (0.1 * i) - (resistance * 0.2) + (volatility * 0.35)
            threshold = max(0.35, min(0.95, adaptive_threshold))

            wavefront_data.append({
                "window": win,
                "pressure": surge,
                "resistance": resistance,
                "momentum": momentum,
                "acceleration": acceleration,
                "distortion": distortion,
                "surge_potential": surge_potential,
                "threshold": threshold
            })

        return wavefront_data

    def early_entry_signal(self):
        wavefront = self.adaptive_fpi_dashboard()

        short = wavefront[0]
        long = wavefront[-1]

        # Detect Pressure Acceleration Over Time
        if len(self.pressure_history) >= 3:
            pressure_slope = self.pressure_history[-1] - self.pressure_history[-3]
        else:
            pressure_slope = 0

        # Resistance Collapse
        resistance_drop = short['resistance'] < (0.7 * long['resistance'])

        # Surge Potential Check
        high_potential = short['surge_potential'] > 0.6 and long['surge_potential'] > 0.5

        # Alien-Level Early Warning
        if (
            pressure_slope > 0.4
            and resistance_drop
            and high_potential
            and short['acceleration'] > 0
            and short['distortion'] < 0.2
        ):
            return "⚡ ALIEN SURGE ACTIVATION IMMINENT"

        return None

def verify_qfe_predictions(df, qfe):
    current_round_count = len(df)
    
    # Process predictions that haven't been checked
    for round_idx, prediction in list(st.session_state.qfe_predictions.items()):
        # Skip if we don't have the next round yet
        if round_idx + 1 >= current_round_count:
            continue
            
        # Get actual outcome
        actual_multiplier = df.iloc[round_idx + 1]['multiplier']
        is_correct = False
        
        # Check prediction accuracy
        if prediction == "SURGE_IMMINENT":
            is_correct = actual_multiplier >= 2.0
        elif prediction == "TRAP_DEPLOYING":
            is_correct = actual_multiplier < 1.5
        
        # Update accuracy stats
        st.session_state.qfe_accuracy['total'] += 1
        if is_correct:
            st.session_state.qfe_accuracy['correct'] += 1
        
        # Update QFE threshold
        qfe.update_threshold(is_correct)
        
        # Remove verified prediction
        del st.session_state.qfe_predictions[round_idx]
    
    # Update last checked index
    st.session_state.qfe_accuracy['last_checked'] = current_round_count - 1

    
    
    st.session_state.qfe_accuracy['last_checked'] = current_round_count - 1

def plot_adaptive_wavefront(wavefront_data):
    fig = go.Figure()

    windows = [w['window'] for w in wavefront_data]

    # 1️⃣ Pressure Wave Bars
    fig.add_trace(go.Bar(
        x=windows,
        y=[w['pressure'] for w in wavefront_data],
        name='Surge Pressure',
        marker_color='royalblue',
        width=0.5,
        opacity=0.85
    ))

    # 2️⃣ Resistance Reef (Area)
    fig.add_trace(go.Scatter(
        x=windows,
        y=[w['resistance'] for w in wavefront_data],
        fill='tozeroy',
        name='Resistance Reef',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='crimson', width=2, dash='solid')
    ))

    # 3️⃣ Adaptive Threshold Tide Line
    fig.add_trace(go.Scatter(
        x=windows,
        y=[w['threshold'] for w in wavefront_data],
        mode='lines+markers',
        name='Adaptive Threshold',
        line=dict(color='gold', dash='dash', width=2),
        marker=dict(symbol='diamond', size=10, color='gold')
    ))

    # 4️⃣ Composite Surge Potential Line
    fig.add_trace(go.Scatter(
        x=windows,
        y=[w['surge_potential'] for w in wavefront_data],
        mode='lines+markers',
        name='Surge Potential',
        line=dict(color='green', width=3),
        marker=dict(symbol='circle', size=9, color='darkgreen')
    ))

    # 5️⃣ Acceleration Overlay
    fig.add_trace(go.Scatter(
        x=windows,
        y=[w['acceleration'] for w in wavefront_data],
        mode='lines+markers',
        name='Phase Acceleration',
        line=dict(color='purple', dash='dot', width=2),
        marker=dict(symbol='x', size=8, color='purple')
    ))

    # 6️⃣ Distortion Energy Fill
    fig.add_trace(go.Scatter(
        x=windows,
        y=[w['distortion'] for w in wavefront_data],
        fill='tozeroy',
        name='Distortion Energy',
        fillcolor='rgba(128,128,128,0.15)',
        line=dict(color='gray', width=1, dash='dot')
    ))

    # 7️⃣ Momentum Annotations (Quiver-like Arrows)
    for w in wavefront_data:
        fig.add_annotation(
            x=w['window'],
            y=w['pressure'] + 0.05,
            ax=0,
            ay=-40 if w['momentum'] < 0 else 40,
            xref='x',
            yref='y',
            axref='pixel',
            ayref='pixel',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='green' if w['momentum'] > 0 else 'red',
            text=f"Momentum: {w['momentum']:.2f}",
            font=dict(color='white', size=10)
        )

    # 8️⃣ Final Layout Tuning
    fig.update_layout(
        title='🌌 Quantum Pressure Wavefront Analysis',
        xaxis_title='Fibonacci Window',
        yaxis_title='Intensity / Score',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

def normalize_msi(msi_series, window_size):
    if len(msi_series) < window_size:
        return 0.5  # Default midpoint if not enough data
    
    window = msi_series[-window_size:]
    min_val, max_val = np.min(window), np.max(window)
    range_width = max_val - min_val
    
    if range_width < 0.01:
        return 0.5  # Flat regime: no volatility
    
    normalized = (msi_series[-1] - min_val) / range_width
    return max(0, min(1, normalized))  # Bound to [0,1]

def dynamic_threshold(range_width):
    if range_width < 0.5:
        return 0.618
    elif range_width < 1.0:
        return 0.5
    else:
        return 0.382

def compute_range_width(msi_series, window_size):
    if len(msi_series) < window_size:
        return 0.0
    window = msi_series[-window_size:]
    return np.max(window) - np.min(window)

def anti_trap_entry_signal(msi_dict):
    """
    msi_dict = {
        3: msi_series_3,
        5: msi_series_5,
        8: msi_series_8
    }
    """
    confirmations = 0

    for window_size, series in msi_dict.items():
        range_width = compute_range_width(series, window_size)
        threshold = dynamic_threshold(range_width)
        norm_msi = normalize_msi(series, window_size)
        
        if norm_msi > threshold:
            confirmations += 1

    if confirmations >= 2:
        return "✅ CLEAN ENTRY ZONE DETECTED"
    else:
        return "⚠️ TRAP RISK - HOLD FIRE"
        
def get_normalized_signal_data(msi_dict):
    data = []
    for window_size, series in msi_dict.items():
        range_width = compute_range_width(series, window_size)
        threshold = dynamic_threshold(range_width)
        norm_msi = normalize_msi(series, window_size)
        
        trap_risk = norm_msi <= threshold
        data.append({
            'Window': window_size,
            'Normalized_MSI': norm_msi,
            'Threshold': threshold,
            'Range_Width': range_width,
            'Trap_Risk': trap_risk
        })
    return pd.DataFrame(data)










# =============== DYNAMIC RANGE REGIME ENGINE ===============
RANGE_WINDOW = 20  # Configurable in sidebar
VOLATILITY_THRESHOLDS = {
    'micro': 1.5,
    'meso': 3.0,
    'macro': 5.0
}

@st.cache_data
def calculate_range_metrics(df, window=20):
    df = df.copy()
    df = df.reset_index(drop=True)
    df['round_index'] = np.arange(len(df))

    # Rolling computations
    df['range_width'] = df['multiplier'].rolling(window, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True
    )
    df['range_center'] = df['multiplier'].rolling(window, min_periods=1).mean()

    # Slopes (difference)
    df['width_slope'] = df['range_width'].diff().fillna(0)
    df['center_slope'] = df['range_center'].diff().fillna(0)

    # Volatility classification
    conditions = [
        (df['range_width'] < VOLATILITY_THRESHOLDS['micro']),
        (df['range_width'] < VOLATILITY_THRESHOLDS['meso']),
        (df['range_width'] >= VOLATILITY_THRESHOLDS['meso'])
    ]
    choices = ['micro', 'meso', 'macro']
    df['volatility_class'] = np.select(conditions, choices, default='unknown')

    # Regime state detection
    df['regime_state'] = np.where(
        (df['width_slope'] > 0) & (df['center_slope'] > 0),
        'surge_favorable',
        np.where(
            (df['width_slope'] < 0) & (df['center_slope'].abs() < 0.1),
            'trap_zone',
            'neutral'
        )
    )

    return df

    
def plot_range_regime(df):
    fig = go.Figure()

    # Range Width
    fig.add_trace(go.Scatter(
        x=df['round_index'],
        y=df['range_width'],
        name='Range Width',
        line=dict(color='royalblue', width=2)
    ))

    # Range Center
    fig.add_trace(go.Scatter(
        x=df['round_index'],
        y=df['range_center'],
        name='Range Center',
        line=dict(color='white', width=2, dash='dot')
    ))

    # Regime Markers
    for regime, color in [('surge_favorable', 'white'), ('trap_zone', 'red')]:
        regime_df = df[df['regime_state'] == regime]
        if not regime_df.empty:
            fig.add_trace(go.Scatter(
                x=regime_df['round_index'],
                y=regime_df['range_center'],
                mode='markers',
                name=regime.upper(),
                marker=dict(color=color, size=10, symbol='diamond')
            ))

    fig.update_layout(
        title='🔥 DYNAMIC RANGE REGIME ENGINE',
        xaxis_title='Round Index',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


# =============== DECISION HUD ===============
def render_regime_hud(current_regime):
    
    if current_regime == 'surge_favorable':
        st.success("✅ FIRE AT WILL")
    
    if current_regime == 'trap_zone':
        st.error("⛔ ABORT ALL ENTRIES")
            
    else:
        st.warning("⚠️ SCOUT ONLY")

    return



def compute_raw_range_signals(
    input_df,
    window=RANGE_WINDOW,
    width_change_window=5,
    slope_windows=[3,5,8,13],
    entropy_bins=10
    ):
    """
    For each round, compute:
        - Range Width
        - Range Center
        - Width Slope
        - Entropy
        - Slope Std (multi-window resonance)
    """
    df = input_df[['timestamp', 'multiplier']].copy()

    # 1. Range Width
    df['range_width'] = df['multiplier'].rolling(window, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True
    )

    # 2. Range Center
    df['range_center'] = df['multiplier'].rolling(window, min_periods=1).mean()

    # 3. Width Slope (smoothed)
    df['width_slope'] = df['range_width'].diff().rolling(width_change_window, min_periods=1).mean()

    # 4. Center Slope
    df['center_slope'] = df['range_center'].diff().rolling(width_change_window, min_periods=1).mean()

    # 5. Entropy
    def calc_entropy(series, bins=entropy_bins):
        counts, _ = np.histogram(series, bins=bins, density=True)
        counts = counts[counts > 0]
        return -np.sum(counts * np.log(counts)) if len(counts) > 0 else 0

    df['entropy'] = df['multiplier'].rolling(window).apply(
        lambda x: calc_entropy(x, bins=entropy_bins), raw=False
    )
    # 6. Slope Resonance Std
    fib_windows = [3, 5, 8, 13]
    slope_cols = []
    for w in fib_windows:
        col = f'slope_{w}'
        df[col] = df['multiplier'].rolling(w, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False)
        
        slope_cols.append(col)
    df['slope_std'] = df[slope_cols].std(axis=1)

    return df


def plot_raw_range_signals(df):
    """Plot the raw range modulation signals with Plotly."""
    if df.empty or len(df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="📊 No trap data yet. Play some rounds to see analysis.",
            showarrow=False,
            font=dict(size=16),
            xref='paper', yref='paper', x=0.5, y=0.5
        )
        fig.update_layout(
            title='🔥 Advanced Range Modulation Analysis',
            xaxis_title='Timestamp',
            yaxis_title='Value',
            hovermode="x unified"
        )
        return fig

    fig = go.Figure()

    # Add traces for each signal
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['range_width'],
        name='Range Width',
        line=dict(color='royalblue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['range_center'],
        name='Range Center',
        line=dict(color='white', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['width_slope'],
        name='Width Slope',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['slope_std'],
        name='Slope Std (Resonance)',
        line=dict(color='red')
    ))

    # Update layout
    fig.update_layout(
        title='📊 Advanced Range Modulation Signals',
        yaxis_title='Metric Value',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig

FIB_WINDOWS = [3, 5, 8, 13, 21, 34]

def compute_multiwindow_atr(df, multiplier_col='multiplier'):
    results = []

    for w in FIB_WINDOWS:
        if len(df) < w + 1:
            continue

        atr_series = df[multiplier_col].rolling(w).apply(lambda x: x.max() - x.min(), raw=True)
        atr_now = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-2] if len(atr_series) > 1 else atr_now

        # Oscillation Phase
        delta_atr = atr_now - atr_prev
        phase = delta_atr / (atr_now + 1e-6)  # avoid divide by zero

        results.append({
            'Window': w,
            'ATR_Now': round(atr_now, 3),
            'ATR_Prev': round(atr_prev, 3),
            'Delta_ATR': round(delta_atr, 3),
            'Phase': round(phase, 4)
        })

    return pd.DataFrame(results)

def detect_phase_regime(oscillator_df):
    # If not enough data, return neutral
    if oscillator_df.empty or len(oscillator_df) < 2:
        return 'NEUTRAL', 0.0

    # Short windows vs long windows
    short_phases = oscillator_df[oscillator_df['Window'] <= 8]['Phase']
    long_phases = oscillator_df[oscillator_df['Window'] >= 13]['Phase']

    if short_phases.empty or long_phases.empty:
        return 'NEUTRAL', 0.0

    # Calculate correlation
    corr = short_phases.corr(long_phases)
    corr = 0.0 if pd.isna(corr) else corr

    # Alien Regime Classification
    if short_phases.mean() > 0 and long_phases.mean() > 0 and corr > 0.3:
        regime = "SURGE_ZONE"
    elif short_phases.mean() > 0 and long_phases.mean() < 0 and corr < -0.2:
        regime = "TRAP_ZONE"
    elif short_phases.mean() < 0 and long_phases.mean() < 0:
        regime = "EXHAUSTION"
    else:
        regime = "NEUTRAL"

    return regime, round(corr, 3)

def plot_atr_oscillator_dashboard(oscillator_df, regime_label, corr_value):
    if oscillator_df.empty:
        st.warning("📊 No ATR Oscillator data to display yet.")
        return

    fig = go.Figure()

    # ATR Now bars
    fig.add_trace(go.Bar(
        x=oscillator_df['Window'],
        y=oscillator_df['ATR_Now'],
        name='ATR Now',
        marker_color='royalblue',
        opacity=0.8
    ))

    # ATR Delta line
    fig.add_trace(go.Scatter(
        x=oscillator_df['Window'],
        y=oscillator_df['Delta_ATR'],
        mode='lines+markers',
        name='Delta ATR',
        line=dict(color='orange', width=2)
    ))

    # Phase line
    fig.add_trace(go.Scatter(
        x=oscillator_df['Window'],
        y=oscillator_df['Phase'],
        mode='lines+markers',
        name='Phase',
        line=dict(color='green', dash='dot')
    ))

    fig.update_layout(
        title=f'🚀 Multi-Window ATR Oscillator - Regime: {regime_label} (Corr={corr_value})',
        xaxis_title='Fibonacci Window Size',
        yaxis_title='ATR / Delta / Phase',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

def compute_atr(series):
    return series.max() - series.min()

def compute_slope(series):
    x = np.arange(len(series))
    if len(series) < 2:
        return 0
    return np.polyfit(x, series, 1)[0]

def compute_phase(series):
    # Upward if last > first, else downward
    return 1 if series.iloc[-1] > series.iloc[0] else -1

def analyze_multi_window_atr_oscillator(
    df,
    multiplier_col='multiplier',
    windows=[3,5,8,13,21,34]
):
    results = []
    
    

    for w in windows:
        if len(df) < w:
            results.append({
                'Window': w,
                'ATR': np.nan,
                #'round_index': np.nan,
                'Slope': np.nan,
                'Phase': np.nan
            })
            continue

        series = df[multiplier_col].tail(w)

        atr = compute_atr(series)
        slope = compute_slope(series)
        phase = compute_phase(series)
        #index_slice = df['round_index'].values[-len(atr_series):]

        results.append({
            'Window': w,
            'ATR': round(atr, 3),
            #'round_index': index_slice,
            'Slope': round(slope, 4),
            'Phase': phase
        })

    df_result = pd.DataFrame(results)

    # Phase alignment score
    phase_alignment = df_result['Phase'].sum() / len(df_result.dropna())

    # Dominant cycle = highest ATR
    dominant_idx = df_result['ATR'].idxmax()
    dominant_window = df_result.loc[dominant_idx, 'Window'] if not np.isnan(dominant_idx) else None

    return df_result, round(phase_alignment, 3), dominant_window

def detect_phase_cross_intersections(df_result):
    """
    Detects where phase directions change between adjacent windows.
    Returns list of window pairs with crossings.
    """
    crossings = []
    phases = df_result['Phase'].values

    for i in range(1, len(phases)):
        if np.isnan(phases[i-1]) or np.isnan(phases[i]):
            continue
        if phases[i] != phases[i-1]:
            crossings.append((df_result['Window'].iloc[i-1], df_result['Window'].iloc[i]))

    return crossings

def plot_multi_window_atr_dashboard(df_result, phase_alignment, dominant_window, crossings):
    fig = go.Figure()

    # ATR Bars
    fig.add_trace(go.Bar(
        x=df_result['Window'],
        y=df_result['ATR'],
        name='ATR (Amplitude)',
        marker_color='purple',
        opacity=0.7
    ))

    # Slope Line
    fig.add_trace(go.Scatter(
        x=df_result['Window'],
        y=df_result['Slope'],
        mode='lines+markers',
        name='Slope (Trend)',
        line=dict(color='orange', dash='dash'),
        marker=dict(size=10)
    ))

    # Phase Markers
    colors = ['green' if p == 1 else 'red' for p in df_result['Phase']]
    fig.add_trace(go.Scatter(
        x=df_result['Window'],
        y=[0]*len(df_result),
        mode='markers',
        marker=dict(color=colors, size=12, symbol='diamond'),
        name='Phase Direction'
    ))

    # Cross-Intersections
    for cross in crossings:
        fig.add_vrect(
            x0=cross[0]-0.5, x1=cross[1]+0.5,
            fillcolor='red', opacity=0.15, line_width=0,
            annotation_text="Phase Cross", annotation_position="top left"
        )

    # Layout
    fig.update_layout(
        title=f"🚀 Multi-Window ATR Oscillator\nPhase Alignment: {phase_alignment} | Dominant Window: {dominant_window}",
        xaxis_title="Fibonacci Window Size",
        yaxis_title="ATR / Slope Values",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig

def compute_smoothed_atr_long_df(df, windows, multiplier_col='multiplier', smooth_window=5, poly_order=2):
    records = []
    df = df.copy()
    df['round_index'] = range(len(df))
    
    for w in windows:
        if len(df) < w:
            continue  # Skip if window larger than data

        # Calculate True Range (max - min) for each window
        atr_series = df[multiplier_col].rolling(w).apply(
            lambda x: x.max() - x.min(), 
            raw=True
        ).bfill().fillna(0)
        
        # Smooth with Savitzky-Golay filter
        if len(atr_series) >= smooth_window:
            atr_smooth = savgol_filter(atr_series, smooth_window, poly_order)
        else:
            atr_smooth = atr_series.values
            
        # Calculate slope and phase
        slope_series = np.gradient(atr_smooth)
        phase_series = np.where(slope_series >= 0, 'BULL', 'BEAR')
        
        # Store results
        for idx, r_idx in enumerate(df['round_index']):
            records.append({
                'round_index': r_idx,
                'window': w,
                'atr': atr_smooth[idx] if idx < len(atr_smooth) else 0,
                'slope': slope_series[idx] if idx < len(slope_series) else 0,
                'phase': phase_series[idx] if idx < len(phase_series) else 'NEUTRAL'
            })
    
    return pd.DataFrame(records)

    #result_df = pd.DataFrame.from_records(records)
    #return result_df

def prepare_long_df_for_pivot(long_df, windows, full_round_index):
    """
    Ensures all round_index/window pairs exist in long_df for pivot.
    Missing atr values are filled with 0.
    """
    grid = pd.MultiIndex.from_product([full_round_index, windows], names=['round_index', 'window']).to_frame(index=False)
    merged = pd.merge(grid, long_df, on=['round_index', 'window'], how='left')
    merged['atr'] = merged['atr'].fillna(0)
    return merged
    
def detect_advanced_crossings(long_df):
    if long_df.empty:
        return []

    pivot = long_df.pivot(index='round_index', columns='window', values='atr').fillna(0)
    crossings = []

    for i in range(1, len(pivot)):
        for w1 in pivot.columns:
            for w2 in pivot.columns:
                if w1 >= w2:
                    continue
                prev_diff = pivot.iloc[i-1][w1] - pivot.iloc[i-1][w2]
                curr_diff = pivot.iloc[i][w1] - pivot.iloc[i][w2]
                if prev_diff * curr_diff < 0:
                    crossings.append({
                        'round_index': pivot.index[i],
                        'window_pair': (w1, w2)
                    })
    return crossings


def detect_smoothed_dominant_window(long_df):
    if long_df.empty:
        return pd.DataFrame()

    pivot = long_df.pivot(index='round_index', columns='window', values='atr').fillna(0)
    dominant = pivot.idxmax(axis=1)

    # Optional smoothing
    dominant_smooth = dominant.rolling(3, min_periods=1).apply(lambda x: x.mode()[0])

    return pd.DataFrame({
        'round_index': pivot.index,
        'dominant_window': dominant_smooth
    })
    

def plot_alien_mwatr_oscillator(long_df, crossings=[]):
    if long_df.empty:
        st.warning("⚠️ Not enough data to plot MWATR Oscillator.")
        return

    fig = go.Figure()
    
    # Color mapping for phases
    phase_colors = {'BULL': '#00ff88', 'BEAR': '#ff0066', 'NEUTRAL': '#888888'}
   
    
    # Plot each Fibonacci window's oscillation
    for w in sorted(long_df['window'].unique()):
        window_df = long_df[long_df['window'] == w]
        
        fig.add_trace(go.Scatter(
            x=window_df['round_index'],
            y=window_df['atr'],
            mode='lines',
            name=f'F{w} Osc',
            line=dict(
                width=2 + w/5 # Thicker line for larger windows
            ),
            hoverinfo='x+y+name',
            showlegend=True
        ))
    
    # Mark crossings with vertical lines
    for cross in crossings:
        fig.add_vline(
            x=cross['round_index'],
            line_dash='dot',
            line_color='gold',
            annotation_text=f"X: F{cross['window_pair'][0]}↔F{cross['window_pair'][1]}", 
            annotation_position='top'
        )
    
    # Layout
    fig.update_layout(
        title='🌌 ALIEN MWATR OSCILLATOR (True Oscillating Range Curves)',
        xaxis_title='Round Index',
        yaxis_title='Range Amplitude',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def compute_quantum_entanglement(long_df):
    """Measures how locked the oscillations are."""
    pivot = long_df.pivot(index='round_index', columns='window', values='slope')
    
    # Calculate sync score (0-1)
    sync_scores = []
    for i in range(len(pivot)):
        window_corr = pivot.iloc[i].rolling(3).corr().mean().mean()  # Multi-window correlation
        sync_scores.append(window_corr)
    
    # Normalize to 0-1 scale
    qei = (np.array(sync_scores) - (-1)) / 2  # Convert [-1,1] to [0,1]
    return qei

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
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
    
@st.cache_data 
def compute_supertrend(df, period=10, multiplier=2.0, source="msi"):
        df = df.copy()
        src = df[source]
        hl2 = src  # substitute for high+low/2
    
        # True range approximation
        df['prev_close'] = src.shift(1)
        df['tr'] = abs(src - df['prev_close'])
        df['atr'] = df['tr'].rolling(window=period).mean()
    
        # Bands
        df['upper_band'] = hl2 - multiplier * df['atr']
        df['lower_band'] = hl2 + multiplier * df['atr']
    
        # Initialize trend
        trend = [1]  # start with uptrend
    
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
    
            upper_band = max(curr['upper_band'], prev['upper_band']) if prev['prev_close'] > prev['upper_band'] else curr['upper_band']
            lower_band = min(curr['lower_band'], prev['lower_band']) if prev['prev_close'] < prev['lower_band'] else curr['lower_band']
    
            if trend[-1] == -1 and curr['prev_close'] > lower_band:
                trend.append(1)
            elif trend[-1] == 1 and curr['prev_close'] < upper_band:
                trend.append(-1)
            else:
                trend.append(trend[-1])
    
            df.at[df.index[i], 'upper_band'] = upper_band
            df.at[df.index[i], 'lower_band'] = lower_band
    
        df["trend"] = trend
        df["supertrend"] = np.where(df["trend"] == 1, df["upper_band"], df["lower_band"])
        df["buy_signal"] = (df["trend"] == 1) & (pd.Series(trend).shift(1) == -1)
        df["sell_signal"] = (df["trend"] == -1) & (pd.Series(trend).shift(1) == 1)
    
        return df
    




    
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





# =================== DATA ANALYSIS ========================
@st.cache_data(show_spinner=False)
def analyze_data(data, pink_threshold, window_size, RANGE_WINDOW, VOLATILITY_THRESHOLDS, window = selected_msi_windows):
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= pink_threshold else ("Purple" if x >= 2 else "Blue"))
    df["msi"] = df["score"].rolling(window_size).sum()
    df["momentum"] = df["score"].cumsum()
    df["round_index"] = range(len(df))
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

    df["rsi"] = compute_rsi(df["score"], period=14)
    
    df["rsi_mid"]   = df["rsi"].rolling(14).mean()
    df["rsi_std"]   = df["rsi"].rolling(14).std()
    df["rsi_upper"] = df["rsi_mid"] + 1.2 * df["rsi_std"]
    df["rsi_lower"] = df["rsi_mid"] - 1.2 * df["rsi_std"]
    df["rsi_signal"] = df["rsi"].ewm(span=7, adjust=False).mean()
    
    # === Ichimoku Cloud on MSI ===
    high_9  = df["msi"].rolling(window=9).max()
    low_9   = df["msi"].rolling(window=9).min()
    df["tenkan"] = (high_9 + low_9) / 2
    
    high_26 = df["msi"].rolling(window=26).max()
    low_26  = df["msi"].rolling(window=26).min()
    df["kijun"] = (high_26 + low_26) / 2
    
    df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    
    high_52 = df["msi"].rolling(window=52).max()
    low_52  = df["msi"].rolling(window=52).min()
    df["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
    
    df["chikou"] = df["msi"].shift(-26)
    df = compute_supertrend(df, period=10, multiplier=2.0, source="msi")

    # Core Fibonacci multipliers
    fib_ratios = [1.0, 1.618, 2.618]
    
    # Center line: rolling MSI mean
    df["feb_center"] = df["msi"].rolling(window=fib_window).mean()
    df["feb_std"] = df["msi"].rolling(window=fib_window).std()
    
    # Upper bands
    df["feb_upper_1"] = df["feb_center"] + fib_ratios[0] * df["feb_std"]
    df["feb_upper_1_618"] = df["feb_center"] + fib_ratios[1] * df["feb_std"]
    df["feb_upper_2_618"] = df["feb_center"] + fib_ratios[2] * df["feb_std"]
    
    # Lower bands
    df["feb_lower_1"] = df["feb_center"] - fib_ratios[0] * df["feb_std"]
    df["feb_lower_1_618"] = df["feb_center"] - fib_ratios[1] * df["feb_std"]
    df["feb_lower_2_618"] = df["feb_center"] - fib_ratios[2] * df["feb_std"]

    for window in selected_msi_windows:
        col_name = f"msi_{window}"
        #df[col_name] = df["score"].rolling(window=window).mean()
        df[col_name] = df["score"].rolling(window=window).sum()
        
        msi_col = f"msi_{window}"
        slope_col = f"slope_{window}"
        df[slope_col] = df[msi_col].diff()

        slope_col = f"slope_{window}"
        sign_col = f"sign_{window}"
        df[sign_col] = df[slope_col].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

    df["msi_resonance"] = df.apply(
    lambda row: compute_resonance(row, selected_msi_windows, window_weights),
    axis=1
    )

   
    
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
            bandwidth_delta = safe_round(latest.get('bandwidth_delta'))  if 'bandwidth_delta' in latest else 0
                
        
    
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
    
            
     # Energy Integrity Score (EIS)
    blues = len(df[df["score"] < 0])
    purples = len(df[(df["score"] == 1.0) | (df["score"] == 1.5)])
    pinks = len(df[df["score"] >= 2.0])
    eis = (purples * 1 + pinks * 2) - blues       
            
            
    
    # Run resonance analysis if we have enough data
    if N >= 10:  # Need at least 10 rounds
        # Run super-powered harmonic scan
        harmonic_waves, resonance_matrix, resonance_score, tension, entropy = multi_harmonic_resonance_analysis(df)
        
        # Predict next 5 rounds
        resonance_forecast_vals = resonance_forecast(harmonic_waves, resonance_matrix) if harmonic_waves else None

     # ===== QUANTUM ENHANCEMENTS =====
    # 1. Range metrics
    df['range_width'] = df['multiplier'].rolling(RANGE_WINDOW).apply(lambda x: x.max() - x.min(), raw=True)
    df['range_center'] = df['multiplier'].rolling(RANGE_WINDOW).mean()
    
    # 2. Regime classification
    #df['regime_state'] = np.where(
     #   (df['range_width'].diff() > 0) & (df['range_center'].diff() > 0),
      #  'surge_favorable',
       # np.where(
        #    (df['range_width'].diff() < 0) & (df['range_center'].diff().abs() < 0.1),
         #   'trap_zone',
          #  'neutral'
        #)
    #)
    
    # 3. Execute quantum analysis
    #quantum = QuantumGambit(df).execute_quantum()
    #df = quantum.df
    #df = calculate_range_metrics(df, window=RANGE_WINDOW)
   
    
    # Return all computed values
    return (df, latest_msi, window_size, recent_df, msi_score, msi_color, latest_tpi, 
            upper_slope, lower_slope, upper_accel, lower_accel, bandwidth, bandwidth_delta, 
            dominant_cycle, current_round_position, wave_label, wave_pct, dom_slope, micro_slope, 
            eis, interference, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, 
            micro_pct, micro_phase_label, micro_freq, dominant_freq, phase, gamma_amplitude, 
            micro_amplitude, micro_phase, micro_cycle_len, micro_position, harmonic_waves, 
            resonance_matrix, resonance_score, tension, entropy, resonance_forecast_vals)


# =================== MSI CHART PLOTTING ========================
def plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, fib_msi_window, fib_lookback_window, spiral_centers=[], window = selected_msi_windows):
    if len(df) < 2:
        st.warning("Need at least 2 rounds to plot MSI chart.")
        return
        
    # MSI with Bollinger Bands
    st.subheader("MSI with Bollinger Bands")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df["timestamp"], df["msi"], label="MSI", color='black')
    if show_bb:
        
        # BB lines
        ax.plot(df["timestamp"], df["bb_upper"], linestyle='--', color='green')
        ax.plot(df["timestamp"], df["bb_lower"], linestyle='--', color='red')
        ax.fill_between(df["timestamp"], df["bb_lower"], df["bb_upper"], color='gray', alpha=0.1)
        ax.plot(df["timestamp"], df["bb_upper_10"], color='#0AEFFF', linestyle='--', label="upperBB", alpha=1.0)
        ax.plot(df["timestamp"], df["bb_lower_10"], color='#0AEFFF', linestyle='--', alpha=1.0)
       
    ax.axhline(0, color='black', linestyle='-.', linewidth=1.6)
    
    # Highlight squeeze
    ax.scatter(df[df["squeeze_flag"]]["timestamp"], df[df["squeeze_flag"]]["msi"], color='purple', label="Squeeze", s=20)
    
    # Highlight breakouts
    ax.scatter(df[df["breakout_up"]]["timestamp"], df[df["breakout_up"]]["msi"], color='lime', label="Breakout ↑", s=20)
    ax.scatter(df[df["breakout_down"]]["timestamp"], df[df["breakout_down"]]["msi"], color='red', label="Breakout ↓", s=20)

    # === Ichimoku Cloud Overlay ===
    if show_ichimoku:
        
        ax.plot(df["timestamp"], df["tenkan"], label="Tenkan-Sen", color='blue', linestyle='-')
        ax.plot(df["timestamp"], df["kijun"], label="Kijun-Sen", color='orange', linestyle='-')
        
        # Cloud fill (Senkou A and B)
        ax.fill_between(df["timestamp"], df["senkou_a"], df["senkou_b"],
                        where=(df["senkou_a"] >= df["senkou_b"]),
                        interpolate=True, color='lightgreen', alpha=0.2, label="Kumo (Bullish)")
        
        ax.fill_between(df["timestamp"], df["senkou_a"], df["senkou_b"],
                        where=(df["senkou_a"] < df["senkou_b"]),
                        interpolate=True, color='red', alpha=0.2, label="Kumo (Bearish)")
    
    # === SuperTrend Line Overlay ===
    if show_supertrend:
        
        ax.plot(df["timestamp"], df["supertrend"], color='lime' if df["trend"].iloc[-1] == 1 else 'red', linewidth=2, label="SuperTrend")
        
        # Buy/Sell markers
        ax.scatter(df[df["buy_signal"]]["timestamp"], df[df["buy_signal"]]["msi"], marker="^", color="green", label="Buy Signal")
        ax.scatter(df[df["sell_signal"]]["timestamp"], df[df["sell_signal"]]["msi"], marker="v", color="red", label="Sell Signal")
            
    for sc in spiral_centers:
        
        ts = pd.to_datetime(sc["timestamp"])  # ensures scalar timestamp
        color = { "Pink": "magenta", "Purple": "orange", "Blue": "blue" }.get(sc["label"], "gray")
    
        ax.axvline(ts, linestyle='--', color=color, alpha=0.6)
        ax.text(ts, df["msi"].max() * 0.9, f"🌀 {sc['label']}", rotation=90,
                fontsize=8, ha='center', va='top', color=color)
        
    if show_ichimoku:
        for echo in spiral_echoes:
            ts = pd.to_datetime(echo["timestamp"])
            label = f"{echo['gap']}-Echo ({echo['source_label']})"
            ax.axvline(ts, linestyle=':', color="maroon", alpha=0.9)
            ax.text(ts, df["msi"].max() * 0.85, label, rotation=90,
                    fontsize=7, ha='center', va='top', color='black')

    if show_fibo:
        
        if show_fibo_bands:

            # Plot center line and key Fibonacci bands
            ax.plot(df["timestamp"], df["feb_center"], linestyle="--", color="gray", linewidth=1.5)
            
            # Upper bands
            ax.plot(df["timestamp"], df["feb_upper_1_618"], linestyle="--", color="blue", linewidth=1.3)
            ax.plot(df["timestamp"], df["feb_upper_2_618"], linestyle="--", color="black", linewidth=1.3)
            
            # Lower bands
            ax.plot(df["timestamp"], df["feb_lower_1_618"], linestyle="--", color="blue", linewidth=1.3)
            ax.plot(df["timestamp"], df["feb_lower_2_618"], linestyle="--", color="black", linewidth=1.3)
            
            # Optional: Light fill between bands for visualization
            ax.fill_between(df["timestamp"], df["feb_lower_1_618"], df["feb_upper_1_618"],
                            color="pink", alpha=0.3, )
        
        for w in true_flp_watchlist:
            if w["target_round"] < len(df):
                ts = pd.to_datetime(df.loc[w["target_round"], "timestamp"])
                ax.axvline(ts, linestyle='--', color='purple', alpha=0.2)
                ax.text(ts, df["msi"].max() * 0.75, f"FLP +{w['gap']}", rotation=90,
                        fontsize=7, ha='center', va='top', color='purple')
                
        for window in selected_msi_windows:
            col_name = f"msi_{window}"
            ax.plot(df["timestamp"], df[col_name],
                    label=f"MSI {window}",
                    linewidth=1.8,
                    alpha=0.9)
        if show_msi_res:
            
            ax2 = ax.twinx()
            ax2.plot(df["timestamp"], df["msi_resonance"], color='purple', linestyle='--', alpha=0.7, label='MSI Resonance')
            ax2.set_ylabel("Resonance Score", color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

        # ---------------------------------
        # MSI FIBONACCI RETRACEMENT OVERLAY
        # ---------------------------------
        if show_fib_retracement:
            fib_msi_column = f"msi_{fib_msi_window}"
            if fib_msi_column in df.columns:
                fib_msi_series = df[fib_msi_column]
                results = calculate_fibonacci_retracements(fib_msi_series, fib_lookback_window)
        
                if results:
                    retrace, ext, high, low = results
        
                    # Plot retracement levels
                    for level, value in retrace.items():
                        ax.axhline(value, color='navy', linestyle=':', alpha=0.8)
                        ax.text(
                            df["timestamp"].iloc[-1], value,
                            f"{level}",
                            fontsize=9, color='navy', va='bottom'
                        )
        
                    # Plot extension levels
                    if show_fib_ext:
                        for level, value in ext.items():
                            ax.axhline(value, color='purple', linestyle=':', alpha=0.3)
                            ax.text(
                                df["timestamp"].iloc[-1], value,
                                f"{level}x",
                                fontsize=7, color='purple', va='top'
                            )    
        if show_multi_fib_analysis:
            msi_col = f"msi_{fib_msi_window}"
            if msi_col in df.columns:
                multi_fib_results = compute_multi_window_fib_retracements(df, msi_col, multi_fib_windows)
                colors = ["navy", "green", "orange", "purple", "red", "brown"]
        
                for idx, (window, result) in enumerate(multi_fib_results.items()):
                    color = colors[idx % len(colors)]
                    for level, value in result["retracements"].items():
                        ax.axhline(value, color=color, linestyle='--', alpha=0.3)
                        ax.text(
                            df["timestamp"].iloc[-1], value,
                            f"{level} (W{window})",
                            fontsize=7, color=color, va='bottom'
                        )
            
    ax.set_title("📊 MSI Volatility Tracker")
    with st.expander("Legend", expanded=False):
        ax.legend()
    plot_slot = st.empty()
    with plot_slot.container():
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
     resonance_matrix, resonance_score, tension, entropy, resonance_forecast_vals) = analyze_data(df, PINK_THRESHOLD, WINDOW_SIZE, RANGE_WINDOW, VOLATILITY_THRESHOLDS)
    
    
    
    
    
    
    spiral_detector = NaturalFibonacciSpiralDetector(df, window_size=selected_window)
    spiral_centers = spiral_detector.detect_spirals()

    spiral_echoes = get_spiral_echoes(spiral_centers, df)
    # Assuming df is your main DataFrame
    max_rounds = len(df)
    
    true_flp_watchlist = project_true_forward_flp(spiral_centers, fib_layers=selected_fib_layers, max_rounds=max_rounds)
    recent_scores = df['multiplier'].tail(34)  # use biggest fib window
    current_msi_values= [df[f"msi_{w}"].iloc[-1] for w in selected_msi_windows]
    current_slopes= [df[f"slope_{w}"].iloc[-1] for w in selected_msi_windows]
    slope_history_series = [df[f"slope_{w}"].tail(5).tolist() for w in selected_msi_windows]

    pink_df = df[df['multiplier'] >= 10.0]
    last_pink_index = pink_df['round_index'].max() if not pink_df.empty else None
    
     # Optional: history of recent gaps between pinks
    pink_rounds = pink_df['round_index'].sort_values().tolist()
    recent_gaps = [pink_rounds[i] - pink_rounds[i-1] for i in range(1, len(pink_rounds))][-5:]

    phase_score= compute_custom_phase_score(
    current_round_index= df['round_index'].iloc[-1],
    last_pink_index= last_pink_index,
    msi_values= current_msi_values,
    slopes= current_slopes,
    window_sizes= selected_msi_windows
    )

    #phase_score['phase_score']

    #phase_score['phase_label']

    regime_result = classify_regime_state(
    current_round_index=df['round_index'].iloc[-1],
    last_pink_index=last_pink_index,
    recent_scores=recent_scores,
    current_msi_values=current_msi_values,
    current_slopes= current_slopes,
    slope_history_series=slope_history_series,
    phase_score=phase_score['phase_score'],
    recent_gap_history=recent_gaps
    )

    alignment_score, gaps = compute_fib_alignment_score(
    df,
    fib_threshold=10.0,
    lookback_window=fib_alignment_window,
    tolerance=fib_alignment_tolerance
    )
    # Store for chart
    st.session_state["alignment_score_history"].append(alignment_score)
    
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
    plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, fib_msi_window, fib_lookback_window,  spiral_centers=spiral_centers)

    # ===== QUANTUM FIBONACCI ENTANGLEMENT DISPLAY =====
    # 1️⃣ QFE Engine Initialization
    # ============================
    multipliers = df['multiplier'].tolist()
    qfe_engine = QuantumFibonacciEntanglement(multipliers)
    
    # ============================
    # 2️⃣ Generate Wavefront Data
    # ============================
    wavefront_data = qfe_engine.adaptive_fpi_dashboard()
    
    

     # ===== QUANTUM WAR ROOM =====
    #st.header("🚀 QUANTUM WAR ROOM")
    
    # 1. Phase space visualization
    #plot_phase_space(df)
    
    # 2. Multiverse simulation
    
    
    # 3. Entropy override
    #if quantum_engine.entropy_state:
     #   st.subheader(f"🧠 ENTROPY OVERRIDE: {quantum_engine.entropy_state}")
     #  if quantum_engine.entropy_state == 'INVERT':
     #       st.error("⚠️ SYSTEM DETECTED PATTERN - INVERT NEXT SIGNAL")
     #   elif quantum_engine.entropy_state == 'TRUST':
     #       st.success("✅ GENUINE RANDOMNESS - TRUST SYSTEM")
    
    # 4. Temporal signatures
    #if quantum_engine.temporal_signatures:
     #   st.subheader("⏱️ TEMPORAL TRAP SIGNATURES")
      #  st.write(f"Trap clusters at seconds: {quantum_engine.temporal_signatures.get('trap_clusters', [])}")

    
    # Assume you have pandas dataframe df with MSI columns:
    # df['msi_3'], df['msi_5'], df['msi_8']
    
    msi_dict = {
        3: df['msi_3'].dropna().tolist(),
        5: df['msi_5'].dropna().tolist(),
        8: df['msi_8'].dropna().tolist(),
    }
    
    #entry_signal = anti_trap_entry_signal(msi_dict)

    #df_signal = get_normalized_signal_data(msi_dict)
    #fig_signal = plot_normalized_signal_dashboard(df_signal)
    
    range_signals_df = compute_raw_range_signals(df, window=RANGE_WINDOW)
    
    # After data analysis:
    #current_regime = df['regime_state'].iloc[-1] if not df.empty else 'neutral'

    # Calculate oscillator
    oscillator_df = compute_multiwindow_atr(df)
    
    # Detect regime
    regime_label, corr_value = detect_phase_regime(oscillator_df)

    # Analyze
    osc_df, phase_alignment, dominant_window = analyze_multi_window_atr_oscillator(
        df,
        multiplier_col='multiplier',
        windows=[3,5,8,13,21,34]
    )
    
    # Cross-intersections
    crossings = detect_phase_cross_intersections(osc_df)
    
    with st.expander("🧭 Multi-Window ATR Oscillator Analysis", expanded=False):
        # Display
        st.subheader("🧭 Multi-Window ATR Oscillator Analysis")
        st.dataframe(osc_df)
    
        # Visualization
        fig_atr = plot_multi_window_atr_dashboard(osc_df, phase_alignment, dominant_window, crossings)
        st.plotly_chart(fig_atr, use_container_width=True)
        
        st.markdown(f"**Phase Alignment Score:** {phase_alignment}")
        st.markdown(f"**Dominant Cycle Window:** {dominant_window}")
        if crossings:
            st.markdown(f"**Phase Cross Intersections:** {crossings}")
    
    
    st.subheader("🌀 ALIEN MWATR OSCILLATOR (Ultra-Mode)")
    
    FIB_WINDOWS = [3, 5, 8, 13, 21,34]
    
    smoothed_atr_df = compute_smoothed_atr_long_df(df, windows=FIB_WINDOWS)

    # Convert the output
    long_df = smoothed_atr_df.copy()
    
    # Ensure pivot safety
    full_round_index = list(range(df.shape[0]))
    long_df_clean = prepare_long_df_for_pivot(long_df, FIB_WINDOWS, full_round_index)
    
    # Detect crossings
    crossings = detect_advanced_crossings(long_df_clean)

    
        
     # Plot
    plot_alien_mwatr_oscillator(long_df_clean, crossings)
    
    # ---- Quantum Entanglement Analysis ----
    st.subheader("🔗 Quantum Entanglement Index")


    # Calculate QEI
    qei_scores = compute_quantum_entanglement(long_df_clean)
    current_qei = qei_scores[-1] if len(qei_scores) > 0 else 0
    with st.expander("🔮 QEI Interpretation Guide"):
        st.markdown("""
        | QEI Score | Regime State          | Trading Action                          |
        |-----------|-----------------------|-----------------------------------------|
        | 0.8-1.0   | Perfect Entanglement  | ⚡ Max long entries, ride full surges    |
        | 0.6-0.8   | Strong Correlation    | ✅ Normal entries, take profits earlier  |
        | 0.4-0.6   | Weak Correlation      | 🟡 Caution - scout positions only        |
        | 0.0-0.4   | Chaos/Transition      | ❌ Avoid entries, prepare for reversal   |
        
        **Pro Tip:** Combine with dominant window:
        - High QEI + F13/F21 dominant = Mega surge coming
        - Low QEI + F3/F5 dominant = Likely fakeout
        """)

    # Add this below the meter
    if len(qei_scores) > 5:  # Only show if we have enough data
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=long_df_clean['round_index'].unique(),
            y=qei_scores,
            mode='lines',
            name='QEI Trend',
            line=dict(color='#00ffff', width=2)
        ))
        trend_fig.update_layout(
            title='QEI Historical Trend',
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    
    #plot_smoothed_atr_oscillator(smoothed_atr_df)
    #long_df_smooth = combine_smoothed_series_to_longform(atr_smooth_dict)
    #dominant_smooth_df = detect_smoothed_dominant_window(smoothed_atr_df)

    
    #crossings = detect_advanced_crossings(long_df )
    
   

    
    #with st.expander("📊 Advanced Range Modulation Signals Over Time", expanded=False):
        #st.subheader("📊 Advanced Trap Modulation Signals Over Time")
        #st.plotly_chart(plot_raw_range_signals(range_signals_df), use_container_width=True)

   
    
    # In main dashboard:
    #st.plotly_chart(plot_range_regime(df), use_container_width=True)
    #render_regime_hud(current_regime)

    with st.expander("🔎 Fibonacci pressure index+ Range Fuckery Modulation", expanded=False):
        # ============================
        # 3️⃣ Plot the Visualization
        # ============================
        plot_adaptive_wavefront(wavefront_data)
        
       
        #st.subheader("🎯 Anti-Trap Signal")
        #st.success(entry_signal if "CLEAN" in entry_signal else entry_signal)

    with st.expander("🔎 Multi-Cycle Detector Results", expanded=False):
       
        
       st.subheader("🎯 Custom Regime Classifier")
       st.markdown(f"**Regime Type:** {regime_result['regime_type']} ({regime_result['estimated_length']} rounds)")
       st.markdown(f"**Phase:** {regime_result['phase_label']} (Score: {regime_result['phase_score']})")
       st.markdown(f"**Round in Regime:** {regime_result['current_round_in_regime']}/{regime_result['estimated_length']}")
       st.markdown(f"**Rounds to Next Shift:** {regime_result['rounds_to_next_shift']}")
       st.markdown(f"**Fibonacci Alignment:** {regime_result['fib_gap_alignment']}")
       st.markdown(f"**Spiral Projections:** {regime_result['spiral_projection_windows']}")

       st.subheader("📊 Fibonacci Alignment Score")
       st.markdown(f"**Score:** {alignment_score}")
       if gaps:
           st.markdown(f"Gaps between pinks: {gaps}")
           
       st.subheader("📈 Alignment Score Trend")

       if len(st.session_state["alignment_score_history"]) >= 2:
           #st.line_chart(st.session_state["alignment_score_history"])
           st.markdown("_Need at least 2 scores to show trend._")
       else:
           st.markdown("_Need at least 2 scores to show trend._")

       if show_multi_fib_analysis and 'multi_fib_results' in locals():
           
           st.sidebar.subheader("🔎 Fib Confluence Zones")
           for window, res in multi_fib_results.items():
               st.sidebar.markdown(f"**Window {window}**")
               for level, value in res["retracements"].items():
                   st.sidebar.markdown(f"{level}: `{value}`")
            
    

   
    

        

            

 
       
        
             
    
    # === SHOW COSINE PHASE PANEL IF ENABLED ===
    
    
    # === SHOW RQCF PANEL IF ENABLED ===
    
    
    
    # === DECISION HUD PANEL ===
    
    
    # === RRQI STATUS ===
    st.metric("🧠 RRQI", rrqi_val, delta="Last 30 rounds")
    if rrqi_val >= 0.3:
        st.success("🔥 Happy Hour Detected — Tactical Entry Zone")
    elif rrqi_val <= -0.2:
        st.error("⚠️ Dead Zone — Avoid Aggressive Entries")
    else:
        st.info("⚖️ Mixed Zone — Scout Cautiously")
    
    # === WAVE ANALYSIS PANEL ===
    
    
    # === BOLLINGER BANDS STATS ===
    with st.expander("💹 Bollinger Bands Stats", expanded=False):
        st.subheader("💹 Bollinger Bands Stats")
        if upper_slope is not None:
            st.metric("Upper Slope", f"{upper_slope}%")
            st.metric("Upper Acceleration", f"{upper_accel}%")
            st.metric("Lower Slope", f"{lower_slope}%")
            st.metric("Lower Acceleration", f"{lower_accel}%")
            st.metric("Bandwidth", f"{bandwidth} Scale (0-20)")
            st.metric("Bandwidth Delta", f"{bandwidth_delta}% shift from last round")
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
    
#quantum_sidebar()
