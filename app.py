import streamlit as st
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import sklearn
import statsmodels

#import pywt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from datetime import timedelta
#import collections
from collections import defaultdict, deque
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from matplotlib import gridspec
#from thre_fused_tdi_module import plot_thre_fused_tdi
#import morlet_phase_enhancement
#from morlet_phase_enhancement import morlet_phase_panel

# ======================= CONFIG ==========================
st.set_page_config(page_title="CYA Quantum Tracker", layout="wide", page_icon="🔥")
st.title("🔥 CYA MOMENTUM TRACKER: v1000 Lite")

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
    show_macd = st.checkbox("💹 MACD", value=True)
    show_msi_res = st.checkbox("❌ MSI RES", value=True)
    
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

@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
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


@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def dynamic_feb_bands(ma_value, phase_score):
    """Phase-weighted dynamic envelope bands."""
    if phase_score >= 0.75:
        mults = ENVELOPE_MULTS
    elif phase_score >= 0.5:
        mults = [0.85, 1.3, 2.0]
    else:
        mults = [0.7, 1.0, 1.5]
    return [round(ma_value * m, 3) for m in mults]


@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def check_envelope_breakouts(msi_value, bands):
    """Flags for each breakout level."""
    return {
        "breakout_1": msi_value >= bands[0],
        "breakout_1_618": msi_value >= bands[1],
        "breakout_2_618": msi_value >= bands[2],
    }
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
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



# ============================
# MSI FIBONACCI RETRACEMENT MODULE
# ============================
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
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

@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def enhanced_quantum_rsi(df, window=10, slope_len=5):
    # 1. Core slope detection from MSI
    msi_slope = df['msi'].diff(slope_len).fillna(0)
    msi_accel = msi_slope.diff().fillna(0)

    # 2. Weight the gain/loss by slope * acceleration
    delta = df['msi'].diff().fillna(0)

    weighted_up = np.where(msi_slope > 0, delta * (1 + msi_accel), 0)
    weighted_down = np.where(msi_slope < 0, -delta * (1 - msi_accel), 0)

    # 3. Rolling average with exponential memory
    up_ewm = pd.Series(weighted_up).ewm(span=window).mean()
    down_ewm = pd.Series(weighted_down).ewm(span=window).mean()

    # 4. RSI-style final calc
    rs = up_ewm / (down_ewm.replace(0, np.nan))
    rsi_like = 100 - (100 / (1 + rs))

    # 5. Smooth final result to remove noise & spikes
    smooth_rsi = rsi_like.ewm(span=3).mean().fillna(0)
    df['eq_rsi'] = smooth_rsi
    return df


@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def enhanced_msi_analysis(df):
    # Calculate base MSI (your existing implementation)
    #df = calculate_msi(df)  
    
    # Add momentum dimensions
    df['msi_slope'] = np.arctan(df['msi'].diff(5))  # Angle in radians
    df['momentum_impulse'] = np.where(
        df['msi_slope'].abs() > np.radians(25),
        df['msi'].diff(3) * 2,  # Amplify strong moves
        df['msi'].diff(3)
    )
    df['score_std'] = df['score'].rolling(5).std().replace(0, np.nan).bfill()
    df['price_slope'] = df['score'].diff(5) / df['score_std']
    # Convergence detector
    df['price_msi_conv'] = df['price_slope']* df['msi_slope']
    
    return df

@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def plot_enhanced_msi(df):
    fig = go.Figure()
    
    # MSI Baseline
    fig.add_trace(go.Scatter(
        x=df.index, y=df['msi'],
        line=dict(color='#888888'),
        name='MSI Baseline'
    ))
    
    # Momentum Slope
    fig.add_trace(go.Scatter(
        x=df.index, y=np.degrees(df['msi_slope']),
        line=dict(color='#00ffff', width=3),
        name='Slope Angle (°)',
        yaxis='y2'
    ))
    
    # Impulse Bars
    fig.add_trace(go.Bar(
        x=df.index, y=df['momentum_impulse'],
        marker_color=np.where(df['momentum_impulse'] > 0, '#00ff88', '#ff0066'),
        name='Impulse Strength',
        opacity=0.5
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title="Slope Angle (°)",
            overlaying='y',
            side='right',
            range=[-45, 45]
        ),
        title='⚡ MSI Momentum Triangulation'
    )
    return fig

def find_momentum_triangles(df, msi_col='msi', order=3, fib_min=0.5, fib_max=1.618, max_gap=30, min_area=0.5):
    from scipy.signal import argrelextrema

    triangles = []
    msi = df[msi_col]
    local_max = argrelextrema(msi.values, np.greater_equal, order=order)[0]
    local_min = argrelextrema(msi.values, np.less_equal, order=order)[0]
    extrema = sorted(np.concatenate([local_max, local_min]))

    for i in range(len(extrema) - 2):
        i1, i2, i3 = extrema[i], extrema[i+1], extrema[i+2]
        x1, x2, x3 = i1, i2, i3
        y1, y2, y3 = msi[i1], msi[i2], msi[i3]

        # Time-gap filter
        if (x3 - x1) > max_gap:
            continue

        # Sides
        a = abs(y2 - y1)
        b = abs(y3 - y2)
        c = abs(y3 - y1)

        # Fibonacci filter
        if a == 0 or b == 0:
            continue
        ratio = b / a
        if not (fib_min <= ratio <= fib_max):
            continue

        # Triangle area filter
        base = (x3 - x1)
        height = max(y1, y2, y3) - min(y1, y2, y3)
        area = 0.5 * base * height
        if area < min_area:
            continue

        # Classification
        if y1 < y2 > y3:
            triangle_type = "descending"
        elif y1 > y2 < y3:
            triangle_type = "ascending"
        else:
            triangle_type = "symmetrical"

        triangles.append({
            'points': (i1, i2, i3),
            'type': triangle_type,
            'area': area
        })

    return triangles


def plot_momentum_triangles_on_ax(ax, df, triangles, msi_col='msi'):
    for tri in triangles:
        i1, i2, i3 = tri['points']
        t_type = tri['type']
        x_vals = [df['timestamp'].iloc[i1], df['timestamp'].iloc[i2], df['timestamp'].iloc[i3]]
        y_vals = [df[msi_col].iloc[i1], df[msi_col].iloc[i2], df[msi_col].iloc[i3]]

        # Color by type
        if t_type == 'ascending':
            color = 'lime'
        elif t_type == 'descending':
            color = 'red'
        else:
            color = 'dodgerblue'

        # Plot triangle
        ax.plot(x_vals + [x_vals[0]], y_vals + [y_vals[0]], color=color, linewidth=2, linestyle='--', alpha=0.9)
        ax.fill(x_vals + [x_vals[0]], y_vals + [y_vals[0]], color=color, alpha=0.15)

        # Labels A, B, C
        ax.text(x_vals[0], y_vals[0], "A", fontsize=9, color=color, fontweight='bold', ha='center', va='bottom')
        ax.text(x_vals[1], y_vals[1], "B", fontsize=9, color=color, fontweight='bold', ha='center', va='bottom')
        ax.text(x_vals[2], y_vals[2], "C", fontsize=9, color=color, fontweight='bold', ha='center', va='bottom')


# MACD over MSI
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def compute_msi_macd(df, msi_col='msi', fast=6, slow=13, signal=5):
    ema_fast = df[msi_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[msi_col].ewm(span=slow, adjust=False).mean()

    df['msi_macd'] = ema_fast - ema_slow
    df['msi_signal'] = df['msi_macd'].ewm(span=signal, adjust=False).mean()
    df['msi_hist'] = df['msi_macd'] - df['msi_signal']
    return df
    
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def compute_signal_macd(df, col='multiplier', fast=6, slow=13, signal=5):
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    out = pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist}, index=df.index)
    return out

def compute_momentum_adaptive_ma(df, msi_col='msi', base_window=10, max_factor=2):
    df = df.copy()

    # Step 1: Momentum diff & normalization
    momentum_strength = df[msi_col].diff().abs().rolling(base_window).mean().fillna(0)
    max_strength = momentum_strength.max()
    if max_strength == 0 or np.isnan(max_strength):
        max_strength = 1  # prevent division by zero
    norm_momentum = (momentum_strength / max_strength).clip(0, 1)

    # Step 2: Adaptive span range
    adaptive_span = base_window / (1 + max_factor * norm_momentum)
    adaptive_span = adaptive_span.clip(lower=3, upper=base_window * 2)

    # Step 3: Dynamic smoothing
    ama = []
    msi_series = df[msi_col].fillna(method='ffill').fillna(0)
    prev = msi_series.iloc[0]
    for i in range(len(msi_series)):
        alpha = 2 / (adaptive_span.iloc[i] + 1)
        prev = alpha * msi_series.iloc[i] + (1 - alpha) * prev
        ama.append(prev)

    df['msi_amma'] = ama
    return df


def detect_wave_points(series, min_distance=3, rel_prominence=0.2):
    """
    Enhanced peak/trough detection with adaptive parameters
    Returns: (peaks, troughs)
    """
    # Calculate adaptive prominence based on series range
    data_range = np.nanmax(series) - np.nanmin(series)
    prominence = rel_prominence * data_range
    
    # Find peaks (high points)
    peaks, _ = find_peaks(
        series,
        distance=min_distance,
        prominence=prominence,
        width=1
    )
    
    # Find troughs (low points)
    troughs, _ = find_peaks(
        -series,
        distance=min_distance,
        prominence=prominence,
        width=1
    )
    
    return peaks, troughs

def label_wave_segments(df, msi_col='msi', min_segment_length=3):
    """
    Identifies wave segments and labels them with direction
    Returns: (df, wave_directions)
    """
    df = df.copy()
    series = df[msi_col]
    
    # Detect wave points with looser parameters
    peaks, troughs = detect_wave_points(series)
    
    # Combine and sort all turning points
    turning_points = sorted(np.concatenate([peaks, troughs]))
    
    # Ensure we have enough points to form segments
    if len(turning_points) < 2:
        df['wave_phase'] = None
        df['wave_label'] = None
        return df, []
    
    # Label wave directions and phases
    wave_directions = []
    for i in range(1, len(turning_points)):
        start_idx = turning_points[i-1]
        end_idx = turning_points[i]
        
        # Skip very small segments
        if (end_idx - start_idx) < min_segment_length:
            continue
            
        direction = 'up' if series[end_idx] > series[start_idx] else 'down'
        wave_directions.append((start_idx, end_idx, direction))
    
    # Create wave labels
    df['wave_phase'] = None
    df['wave_label'] = None
    
    for i, (start, end, direction) in enumerate(wave_directions):
        df.loc[start:end, 'wave_phase'] = direction
        df.loc[start, 'wave_label'] = f"W{i+1}S"
        df.loc[end, 'wave_label'] = f"W{i+1}E"
    
    return df, wave_directions

def assign_elliott_waves(df, wave_directions):
    """
    Assigns Elliot Wave labels (1-5 for impulse, A-B-C for corrective)
    Returns: df with 'elliott_wave' column
    """
    df = df.copy()
    df['elliott_wave'] = None
    
    if not wave_directions:
        return df
    
    # Determine if we're starting with impulse or corrective
    initial_direction = wave_directions[0][2]
    is_impulse = initial_direction == 'up'
    
    wave_count = 1
    max_waves = 5 if is_impulse else 3  # 1-5 or A-C
    
    for start, end, direction in wave_directions:
        if wave_count > max_waves:
            break
            
        # Assign wave label
        if is_impulse:
            label = str(wave_count)
        else:
            label = chr(64 + wave_count)  # A, B, C
            
        df.loc[start:end, 'elliott_wave'] = f"EW{label}"
        
        wave_count += 1
        
        # Switch to corrective after impulse
        if is_impulse and wave_count > 5:
            is_impulse = False
            wave_count = 1
    
    return df

def plot_wave_labels(ax, df, label_col='wave_label', value_col='msi', time_col='timestamp'):
    """Plots wave labels on chart using timestamp for x-axis"""
    if label_col not in df.columns:
        return
        
    labeled_points = df[df[label_col].notna()]
    
    for _, row in labeled_points.iterrows():
        ax.annotate(
            row[label_col],
            (row[time_col], row[value_col]),  # Use timestamp for x
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            color='blue',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
        )

def plot_elliott_waves(ax, df, wave_col='elliott_wave', value_col='msi', time_col='timestamp'):
    """Highlights Elliot Wave segments using timestamp for x-axis"""
    if wave_col not in df.columns:
        return
        
    # Get unique waves
    waves = df[wave_col].dropna().unique()
    
    for wave in waves:
        wave_df = df[df[wave_col] == wave]
        if len(wave_df) < 2:
            continue
            
        color = 'green' if '1' in wave or '3' in wave or '5' in wave else (
               'red' if 'A' in wave or 'C' in wave else 'blue')
        
        ax.plot(
            wave_df[time_col],  # Use timestamp for x
            wave_df[value_col],
            color=color,
            linewidth=2,
            alpha=0.7,
            label=f"{wave}"
        )

def map_multiplier_level(value):
    if value < 2:
        return 1  # Level 1x
    elif value < 10:
        return 2  # Level 2x
    else:
        return 3  # Level 10x+

@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def plot_multiplier_timeseries(df, multiplier_col='multiplier', time_col='timestamp'):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw multiplier values
    ax.plot(df[time_col], df[multiplier_col], label='Multiplier', color='royalblue', linewidth=1.5)

    # Optional: horizontal lines for visual thresholds
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(2.0, color='orange', linestyle='--', linewidth=1)
    ax.axhline(10.0, color='deeppink', linestyle='--', linewidth=1)

    # Format
    ax.set_title('📈 Multiplier Time Series', fontsize=16)
    ax.set_xlabel('Time')
    ax.set_ylabel('Multiplier')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig

# Safe STL: try to use statsmodels.STL, otherwise fallback to rolling mean seasonal/trend
def stl_or_fallback(signal_array, period=6):
    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(signal_array, period=max(3, int(period)), robust=True)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception:
        # Fallback: trend = rolling mean; seasonal = signal - trend; resid small
        s = pd.Series(signal_array)
        trend = s.rolling(window=max(3, int(period)), min_periods=1, center=True).mean().values
        seasonal = signal_array - trend
        resid = signal_array - trend - seasonal * 0  # basically zeros, kept for shape
        return trend, seasonal, resid

# Safe ACF: adaptive nlags and returns lags with strong spikes
def acf_spike_detector(signal_array, max_lags=20, threshold=0.4):
    n = len(signal_array)
    if n < 4:
        return [] , np.array([])  # not enough data
    nlags = min(max_lags, max(3, n // 2))
    try:
        acf_vals = stats_acf(signal_array, nlags=nlags, fft=True)
    except Exception:
        # fallback simple serial correlation (very small)
        acf_vals = np.array([1.0] + [0.0]*(nlags))
    spike_lags = [i for i,v in enumerate(acf_vals[1:], start=1) if v >= threshold]
    return spike_lags, acf_vals

# Input: minute_avg_df with index or column 'minute' and column 'multiplier'
def build_responsive_signal(minute_avg_df):
    # Ensure sorted and continuous index
    minute_avg_df = minute_avg_df.sort_values('minute').reset_index(drop=True)
    raw = minute_avg_df['multiplier'].values
    N = len(raw)
    if N == 0:
        return np.array([]), minute_avg_df
    # choose window length (odd and <= N)
    wl = 5 if N >= 5 else (N if N%2==1 else max(1, N-1))
    try:
        filtered = savgol_filter(raw, window_length=wl, polyorder=2)
    except Exception:
        # fallback simple rolling mean
        filtered = pd.Series(raw).rolling(window=wl, min_periods=1, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    return filtered, minute_avg_df

def get_top_fft_periods(signal_array, sample_seconds=60, topk=3, min_period_minutes=2, max_period_minutes=120):
    N = len(signal_array)
    if N < 3:
        return []  # nothing meaningful
    # Use rfft for real signals
    yf = rfft(signal_array)
    xf = rfftfreq(N, sample_seconds)  # frequencies in Hz (cycles/sec)
    mag = np.abs(yf)
    # exclude DC (index 0)
    if len(mag) <= 1:
        return []
    mag_nozero = mag.copy()
    mag_nozero[0] = 0
    # pick topk indices but guard duplicates/near-zero freqs
    idx_sorted = np.argsort(mag_nozero)[::-1]
    periods = []
    for idx in idx_sorted:
        if idx == 0:
            continue
        freq = xf[idx]
        if freq <= 0:
            continue
        period_min = 1.0 / freq / 60.0
        if not (min_period_minutes <= period_min <= max_period_minutes):
            continue
        periods.append((period_min, mag_nozero[idx], freq))
        if len(periods) >= topk:
            break
    return periods  # list of tuples (period_minutes, magnitude, freq_hz)

# sine with known angular frequency
def sine_with_fixed_omega(t, A, phi, offset, omega):
    return A * np.sin(omega * t + phi) + offset

def fit_sine_fixed_freq(time_vector, signal_array, freq_hz):
    omega = 2 * math.pi * freq_hz
    # initial guesses
    A0 = (np.nanmax(signal_array) - np.nanmin(signal_array)) / 2.0
    phi0 = 0.0
    offset0 = np.nanmean(signal_array)
    try:
        params, _ = curve_fit(lambda t, A, phi, offset: sine_with_fixed_omega(t, A, phi, offset, omega),
                              time_vector, signal_array, p0=[A0, phi0, offset0], maxfev=2000)
        return params, omega
    except Exception:
        return (A0, phi0, offset0), omega

# Build multi-sine predicted wave from top periods
def build_multi_sine(time_vector, signal_array, top_periods):
    # top_periods: list of (period_min, magnitude, freq_hz)
    recon = np.zeros_like(signal_array, dtype=float)
    fitted_params = []
    for period_min, mag, freq_hz in top_periods:
        params, omega = fit_sine_fixed_freq(time_vector, signal_array, freq_hz)
        A, phi, offset = params
        recon += sine_with_fixed_omega(time_vector, A, phi, offset, omega)  # sum
        fitted_params.append({'period_min': period_min, 'A': A, 'phi': phi, 'offset': offset, 'omega': omega})
    # Normalize by number of components to keep scale sensible (optional)
    if len(top_periods) > 0:
        recon = recon / max(1, len(top_periods))
    return recon, fitted_params
    
@st.cache_data
def predict_future_peaks(minute_index_series, fitted_params, horizon_minutes=60, n_peaks=3):
    # minute_index_series: pandas Series or index of minute timestamps
    # fitted_params: list of fitted sine dicts (A,phi,offset,omega)
    if len(minute_index_series) == 0 or len(fitted_params) == 0:
        return []
    # create time vector in minutes since start
    t0 = pd.to_datetime(minute_index_series.iloc[0])
    last_min = pd.to_datetime(minute_index_series.iloc[-1])
    total_minutes = int((last_min - t0).total_seconds() / 60)
    # future time grid (in minutes since start)
    future_minutes = np.arange(total_minutes, total_minutes + horizon_minutes + 1)
    # build multi-sine forecast using fitted_params
    forecast = np.zeros_like(future_minutes, dtype=float)
    for p in fitted_params:
        omega = p['omega']
        A, phi, offset = p['A'], p['phi'], p['offset']
        t_seconds = future_minutes * 60.0
        forecast += sine_with_fixed_omega(t_seconds, A, phi, offset, omega)
    if len(fitted_params) > 0:
        forecast = forecast / len(fitted_params)
    # detect peaks on forecast
    sec_deriv = np.diff(np.sign(np.diff(forecast)))
    peak_idx = np.where(sec_deriv == -2)[0] + 1
    # convert peak_idx in future_minutes to absolute timestamps
    peak_minutes = future_minutes[peak_idx]
    peak_times = [t0 + pd.Timedelta(minutes=int(m)) for m in peak_minutes]
    # return next n_peaks
    return peak_times[:n_peaks], forecast, future_minutes

def multi_wave_trap_scanner(round_df, windows=[1, 3, 5, 10]):
    """Builds smoothed avg multiplier waves for multiple higher-minute windows."""
    fig, ax = plt.subplots(figsize=(12, 5))
    peak_dict, trough_dict = {}, {}

    for w in windows:
        # Aggregate to higher minute frames
        df_w = (
            round_df.resample(f"{w}T", on="timestamp")['multiplier']
            .mean()
            .reset_index()
            .rename(columns={'multiplier': 'multiplier', 'timestamp': 'minute'})
        )

        signal, df_w = build_responsive_signal(df_w)

        # Peak/trough detection
        peaks, _ = find_peaks(signal, distance=2)
        troughs, _ = find_peaks(-signal, distance=2)

        peak_dict[w] = (df_w['minute'].iloc[peaks], signal[peaks])
        trough_dict[w] = (df_w['minute'].iloc[troughs], signal[troughs])

        # Plot
        ax.plot(df_w['minute'], signal, label=f"{w}-min Wave")
        ax.scatter(df_w['minute'].iloc[peaks], signal[peaks], color='red', marker='o', s=40)
        ax.scatter(df_w['minute'].iloc[troughs], signal[troughs], color='purple', marker='x', s=40)

    ax.set_title("🔮 Multi-Wave Trap Scanner (Smoothed Higher Minute Waves)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    return peak_dict, trough_dict
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
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def analyze_data(data, pink_threshold, window_size, RANGE_WINDOW,  window = selected_msi_windows):
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Round timestamp to the nearest minute
    df['minute'] = df['timestamp'].dt.floor('min')
    
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

    #df = enhanced_msi_analysis(df)
    df = compute_momentum_adaptive_ma(df)
    df = compute_msi_macd(df, msi_col='msi')

    # Multi-window BBs on MSI
    df["bb_mid_20"], df["bb_upper_20"], df["bb_lower_20"] = bollinger_bands(df["msi"], 20, 2)
    df["bb_mid_10"], df["bb_upper_10"], df["bb_lower_10"] = bollinger_bands(df["msi"], 10, 1.5)
    #df["bb_mid_40"], df["bb_upper_40"], df["bb_lower_40"] = bollinger_bands(df["msi"], 40, 2.5)
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

    

    # === Ichimoku Cloud on MSI ===
    high_9  = df["msi"].rolling(window=9).max()
    low_9   = df["msi"].rolling(window=9).min()
    df["tenkan"] = (high_9 + low_9) / 2
    
    high_26 = df["msi"].rolling(window=26).max()
    low_26  = df["msi"].rolling(window=26).min()
    df["kijun"] = (high_26 + low_26) / 2

    high_3 = df["msi"].rolling(3).max()
    low_3 = df["msi"].rolling(3).min()
    df["mini_tenkan"] = (high_3 + low_3)/2

    high_5 = df["msi"].rolling(5).max()
    low_5 = df["msi"].rolling(5).min()
    df["mini_kijun"] = (high_5 + low_5)/2

    #high_2 = df["msi"].rolling(1).max()
    #low_2 = df["msi"].rolling(1).min()
    #df["nano_tenkan"] = df["msi"].ewm(span=2).mean()

    # Projected Senkou A — mini average of short-term structure
    df["mini_senkou_a"] = ((df["mini_tenkan"] + df["mini_kijun"]) / 2).shift(6)
    
    # Projected Senkou B — mini-range memory, 12-period HL midpoint
    high_12 = df["msi"].rolling(12).max()
    low_12 = df["msi"].rolling(12).min()
    df["mini_senkou_b"] = ((high_12 + low_12) / 2).shift(6)

    #df["rsi"] = compute_rsi(df["bb_mid_10"], period=14)
    #df = enhanced_quantum_rsi(df)

    
    #df["rsi_mid"]   =  df['eq_rsi'].rolling(14).mean()
    #df["rsi_std"]   =  df['eq_rsi'].rolling(14).std()
    #df["rsi_upper"] = df["rsi_mid"] + 1.2 * df["rsi_std"]
    #df["rsi_lower"] = df["rsi_mid"] - 1.2 * df["rsi_std"]
    #df["rsi_signal"] =  df['eq_rsi'].ewm(span=7, adjust=False).mean()

    #high_3 = df['eq_rsi'].rolling(3).max()
    #low_3 = df['eq_rsi'].rolling(3).min()
    #df["mini_tenkan_rsi"] = (high_3 + low_3)/2

    #high_5 = df['eq_rsi'].rolling(5).max()
    #low_5 = df['eq_rsi'].rolling(5).min()
    #df["mini_kijun_rsi"] = (high_5 + low_5)/2

    #df["mini_senkou_a_rsi"] = ((df["mini_tenkan_rsi"] + df["mini_kijun_rsi"]) / 2).shift(6)
    
    # Projected Senkou B — mini-range memory, 12-period HL midpoint
    #high_12 = df['eq_rsi'].rolling(12).max()
    #low_12 = df['eq_rsi'].rolling(12).min()
    #df["mini_senkou_b_rsi"] = ((high_12 + low_12) / 2).shift(6)
    

     # MSI[5] and MSI[10]
    #df['msi_5'] = df['multiplier'].rolling(5).mean()
    #df['msi_10'] = df['multiplier'].rolling(10).mean()
    
    # Cross states
    #df['mini_surge'] = (df['msi_5'] > df['tenkan']) & (df['msi_5'].shift(1) <= df['tenkan'].shift(1))
    #df['main_surge'] = (df['msi_10'] > df['tenkan']) & (df['msi_10'].shift(1) <= df['tenkan'].shift(1))
    
    #df['tenkan_angle'] = df['tenkan'].diff()
    #df["tenkan_surge"] = df["tenkan_angle"].abs() > df["tenkan_angle"].rolling(10).std()
    
    # Flat states
    #df['tenkan_flat'] = df['tenkan'].diff().abs() < 1e-6
    #df["flat_zone"] = df["tenkan_flat"].rolling(5).sum() >= 3  # ≥5 consecutive flats
    
    #df['kijun_flat'] = df['kijun'].diff().abs() < 1e-6

    # Tight proximity detection
    #df['tight_gap'] = (df['tenkan'] - df['kijun']).abs() < 0.
    
    #df['trap_zone'] = df['tenkan_flat'] & df['kijun_flat']
    
    df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    
    high_52 = df["msi"].rolling(window=52).max()
    low_52  = df["msi"].rolling(window=52).min()
    df["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
    
    df["chikou"] = df["msi"].shift(-26)
    df = compute_supertrend(df, period=10, multiplier=2.0, source="msi")

    # Custom Stochastic Mini-Momentum Index (SMMI)
    #lowest = df["momentum_impulse"].rolling(5).min()
    #highest = df["momentum_impulse"].rolling(5).max()
    #df["smmi"] = 100 * ((df["momentum_impulse"] - lowest) / (highest - lowest))


    # Core Fibonacci multipliers
    fib_ratios = [1.0, 1.618, 2.618]
    
    # Center line: rolling MSI mean
    #df["feb_center"] = df["msi"].rolling(window=fib_window).mean()
    #df["feb_std"] = df["msi"].rolling(window=fib_window).std()
    
    # Upper bands
    #df["feb_upper_1"] = df["feb_center"] + fib_ratios[0] * df["feb_std"]
    #df["feb_upper_1_618"] = df["feb_center"] + fib_ratios[1] * df["feb_std"]
    #df["feb_upper_2_618"] = df["feb_center"] + fib_ratios[2] * df["feb_std"]
    
    # Lower bands
    #df["feb_lower_1"] = df["feb_center"] - fib_ratios[0] * df["feb_std"]
    #df["feb_lower_1_618"] = df["feb_center"] - fib_ratios[1] * df["feb_std"]
    #df["feb_lower_2_618"] = df["feb_center"] - fib_ratios[2] * df["feb_std"]

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
    #df['range_width'] = df['multiplier'].rolling(RANGE_WINDOW).apply(lambda x: x.max() - x.min(), raw=True)
    #df['range_center'] = df['multiplier'].rolling(RANGE_WINDOW).mean()
    
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
@st.cache_data
@st.cache_data(ttl=600, show_spinner=False)
def plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, fib_msi_window, fib_lookback_window, spiral_centers=[], window = selected_msi_windows):
    if len(df) < 2:
        st.warning("Need at least 2 rounds to plot MSI chart.")
        return

    
    # MSI with Bollinger Bands
    st.subheader("MSI with Bollinger Bands")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df["timestamp"], df["msi"], label="MSI", color='black')

    # ===== Enhanced Wave Detection and Labeling =====
    # Detect and label waves
    df, wave_directions = label_wave_segments(df)
    df = assign_elliott_waves(df, wave_directions)
    
    # ✅ Plot Adaptive Moving Average of MSI (msi_amma)
    if 'msi_amma' in df.columns:
        ax.plot(df["timestamp"], df['msi_amma'], label='MSI-AMMA', color='deepskyblue', linewidth=1.3, linestyle='--')
        
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
        ax.plot(df["timestamp"], df["mini_tenkan"],  label="Mini-tenkan", color='black', linestyle='-', linewidth= 2)
        ax.plot(df["timestamp"], df["nano_tenkan"],  label="nano-tenkan", color='green', linestyle='--', alpha= 0.9)
        
        
        
        #ax.scatter(df[df['main_surge']]["timestamp"], df[df['main_surge']]["msi"], 
           #color="cyan", s=30, marker="*", label="Main Surge")

        #ax.scatter(df[df['mini_surge']]["timestamp"], df[df['mini_surge']]["msi"], 
           #color="green", s=30, marker="*", label="quantum spark")
        
        #for idx in df[df["flat_zone"]].index:
            #ax.axvspan(df["timestamp"].iloc[idx], df["timestamp"].iloc[min(idx + 1, len(df) - 1)],
                       #color='gray', alpha=0.1)
            
        #ax.scatter(df[df["trap_zone"]]["timestamp"], df[df["trap_zone"]]["msi"], 
           #color="orange", s=25, label="Trap Zone")

        # Cloud fill (Mini Senkou A and B)
        ax.fill_between(df["timestamp"], df["mini_senkou_a"], df["mini_senkou_b"],
                        where=(df["mini_senkou_a"] >= df["mini_senkou_b"]),
                        interpolate=True, color='lightgreen', alpha=0.2, label="Kumo (Bullish)")
        
        ax.fill_between(df["timestamp"], df["mini_senkou_a"], df["mini_senkou_b"],
                        where=(df["mini_senkou_a"] < df["mini_senkou_b"]),
                        interpolate=True, color='red', alpha=0.2, label="Kumo (Bearish)")
        
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
        
    if show_supertrend:
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
                    color= 'purple',
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
    # === HARMONIC TRIANGLE OVERLAY ===
    try:
        triangles = find_momentum_triangles(df)
        medium_triangles = find_momentum_triangles(df, msi_col='msi', order=8)
        large_triangles = find_momentum_triangles(df, msi_col='msi', order=13)
        
        plot_momentum_triangles_on_ax(ax, df, triangles)

        plot_momentum_triangles_on_ax(ax, df, medium_triangles)
        plot_momentum_triangles_on_ax(ax, df, large_triangles)

    

    except Exception as e:
        print(f"[Triangle Plot Error] {e}")
        
    # ===== Add These Plotting Calls =====
    plot_wave_labels(ax, df)  # Basic wave labels
    plot_elliott_waves(ax, df)  # Elliot wave highlights
         
    ax.set_title("📊 MSI Volatility Tracker")
    ax.legend()
   
    
    
    # AX3: MACD over MSI
    if show_macd and 'msi_macd' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 2.5))
        ax3.plot(df['msi_macd'], label='MSI-MACD', color='blue')
        ax3.plot(df['msi_signal'], label='MACD Signal', color='red', linestyle='--')
        ax3.bar(df.index, df['msi_hist'], label='MACD Hist', color='gray', alpha=0.4, width=1)
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_ylabel('MSI-MACD')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.15)
    
    #ax3.plot(df.index, df['price_msi_conv'], label='Price-MSI Convergence', color='gold')
    #ax3.axhline(0, linestyle='--', color='gray')
    #ax3.fill_between(df.index, 0, df['price_msi_conv'], where=(df['price_msi_conv'] > 0), color='lime', alpha=0.2)
    #ax3.fill_between(df.index, 0, df['price_msi_conv'], where=(df['price_msi_conv'] < 0), color='red', alpha=0.2)
    #ax3.legend(loc="upper left", fontsize=8)

    plot_slot = st.empty()
    with plot_slot.container():
        st.pyplot(fig)
        #st.pyplot(fig2)
        st.pyplot(fig3)
        
            

# =================== MAIN APP FUNCTIONALITY ========================
# =================== FLOATING ADD ROUND UI ========================


# Fast entry mode UI - simplified UI for mobile/quick decisions

def fast_entry_mode_ui():
    st.markdown("### ⚡ FAST ENTRY MODE")
    st.markdown("Tap a number to enter the corresponding round multiplier")

    # Numberpad layout
    num_rows = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 10]
    ]

    for row in num_rows:
        cols = st.columns(len(row))
        for i, num in enumerate(row):
            with cols[i]:
                if st.button(f"{num}x", use_container_width=True):
                    # Categorize score
                    if num < 2:
                        score = -1  # Blue
                    elif num < 10:
                        score = 1   # Purple
                    else:
                        score = 2   # Pink

                    # Append to roundsc
                    st.session_state.roundsc.append({
                        "timestamp": datetime.now(),
                        "multiplier": float(num),
                        "score": score
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
    fast_entry_mode_ui()

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
     resonance_matrix, resonance_score, tension, entropy, resonance_forecast_vals) = analyze_data(df, PINK_THRESHOLD, WINDOW_SIZE, RANGE_WINDOW)
    
    

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

    
    current_round_index= df['round_index'].iloc[-1],
    
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
    #plot_msi_chart(df, window_size, recent_df, msi_score, msi_color, harmonic_wave, micro_wave, harmonic_forecast, forecast_times, fib_msi_window, fib_lookback_window,  spiral_centers=spiral_centers)
    
    #fig = plot_enhanced_msi(df)
    #st.plotly_chart(fig, use_container_width=True)
    #df['multiplier_level'] = df['multiplier'].apply(map_multiplier_level)

    # Ensure timestamp is parsed
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with bad timestamps
    df = df.dropna(subset=['timestamp'])
    
    
    
    # Round timestamp to the nearest minute
    df['minute'] = df['timestamp'].dt.floor('min')
    
    # Group by each minute and calculate average multiplier
    minute_avg_df = df.groupby('minute').agg({'multiplier': 'mean'}).reset_index()
    
    # Optional: Fill missing minutes if gaps exist (important for clean FFT)
    minute_avg_df.set_index('minute', inplace=True)
    minute_avg_df = minute_avg_df.resample('1min').mean().interpolate()
    minute_avg_df.reset_index(inplace=True)
    minute_avg_df = minute_avg_df.dropna(subset=['multiplier'])

    # DEBUG: Print how many minutes we're working with
    st.write(f"📊 Minutes of data for FFT: {len(minute_avg_df)}")

    # Extract signal: average multiplier values
    signal = minute_avg_df['multiplier'].values
    N = len(signal)
    
    # Guard against too-short series
    if N < 6:
        st.warning(f"⚠️ FFT running on low data ({N} points). Accuracy may be poor.")

    if N < 16:
        signal = savgol_filter(signal, window_length=5 if N >= 5 else N, polyorder=2)

    signal = savgol_filter(signal, window_length=5 if N >= 5 else N, polyorder=2)

    # Ensure timestamp is parsed
    #df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    #df = df.dropna(subset=['timestamp'])
    
    # Round to nearest second (for consistent time axis)
    df['second'] = df['timestamp'].dt.floor('s')
    
    # Group by each second → average multiplier
    sec_df = df.groupby('second').agg({'multiplier': 'mean'}).reset_index()
    
    # Fill missing seconds (important for smooth EWM spans)
    sec_df.set_index('second', inplace=True)
    sec_df = sec_df.resample('1S').mean().interpolate()
    sec_df.reset_index(inplace=True)
    
    # Build clean second-level series
    sec_series = sec_df.set_index('second')['multiplier']
    
    # Fibonacci EWM lines in *seconds*
    fib_spans = [34, 55, 91]
    fib_df = pd.DataFrame({'time': sec_series.index, 'multiplier_sec': sec_series})
    for s in fib_spans:
        fib_df[f'fib{s}'] = sec_series.ewm(span=s, adjust=False).mean().values
        fib_df[f'fib{s}']= savgol_filter(fib_df[f'fib{s}'], window_length=5 if N >= 5 else N, polyorder=2)
    
  
    
    # ========== PLOT ==========
    with st.expander("⏱️ Fibonacci Time Map (34s / 55s / 91s)", expanded=False):
        fig_fib, ax_fib = plt.subplots(figsize=(12, 4))
        ax_fib.plot(fib_df['time'], fib_df['multiplier_sec'], label='Multiplier (1s)', alpha=0.35)
        ax_fib.plot(fib_df['time'], fib_df['fib34'], label='fib34', linewidth=1.2)
        ax_fib.plot(fib_df['time'], fib_df['fib55'], label='fib55', linewidth=1.2)
        ax_fib.plot(fib_df['time'], fib_df['fib91'], label='fib91', linewidth=1.2)
        ax_fib.set_title("Fibonacci-Timed Signal Lines (Second-Level)")
        ax_fib.legend(loc='upper left')
        plt.tight_layout()
        #st.pyplot(fig_fib)
    
        

    
    
    
    # Sample spacing (1 minute interval = 60 seconds)
    T = 60.0  # seconds per sample (1 per minute)
    time = np.arange(N)  # N = number of minutes
    
        # Apply FFT
    yf = rfft(signal)
    xf = rfftfreq(N, T)[:N // 2]  # frequency axis (positive half)
        
    # Magnitude of FFT
    fft_magnitude = 2.0 / N * np.abs(yf[0:N // 2])
    
    # Safety: Check if there's anything to analyze
    if len(fft_magnitude[1:]) == 0:
        raise ValueError("🚫 FFT magnitude array is empty. Check your data input.")
        
    # Get dominant frequency (excluding 0 Hz / DC component)
    dominant_index = np.argmax(fft_magnitude[1:]) + 1
    dominant_freq = xf[dominant_index]  # cycles per second (Hz)
    omega = 2 * np.pi * dominant_freq  # angular frequency

    # Define sine wave function: A * sin(ωt + φ) + offset
    def sine_model(t, A, phi, offset):
        return A * np.sin(omega * t + phi) + offset
        
    # Fit sine wave to the signal using curve fitting
    params, _ = curve_fit(sine_model, time, signal, p0=[1, 0, np.mean(signal)])

   
    
    macd_df   = compute_signal_macd(minute_avg_df.set_index('minute'))
    
    # --- STL Decomposition ---
    # Assume signal is the SavGol-filtered curve
    stl = STL(signal, period=6, robust=True)
    res = stl.fit()
    
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    # Append to your working DataFrame for plotting
    minute_avg_df['stl_trend'] = trend
    minute_avg_df['stl_seasonal'] = seasonal
    minute_avg_df['stl_residual'] = residual

    # ---- assume minute_avg_df exists and has 'minute' and 'multiplier' ----
    filtered_signal, minute_avg_df = build_responsive_signal(minute_avg_df)
    if len(filtered_signal) == 0:
        st.info("No data for time-series predictor yet.")
    else:
        # time vector for fitting (seconds since start)
        N = len(filtered_signal)
        t_seconds = np.arange(N) * 60.0  # each sample = 1 minute -> 60 sec steps
    
        # STL decomposition (period guess from small window)
        est_period = max(3, min(12, int(max(3, N//6))))
        tren, seasona, resid = stl_or_fallback(filtered_signal, period=est_period)
        minute_avg_df['trend'] = tren
        minute_avg_df['seasonal'] = seasona
        minute_avg_df['resid'] = resid
    
        # ACF spike detection (on seasonal)
        spike_lags, acf_vals = acf_spike_detector(seasona, max_lags=min(20, max(5, N//2)), threshold=0.4)
    
        # FFT top periods (work with filtered signal)
        top_periods = get_top_fft_periods(filtered_signal, sample_seconds=60.0, topk=3,
                                          min_period_minutes=2, max_period_minutes=120)
        # extract just freq hz for fitting
        top_periods_simple = [(p[0], p[1], p[2]) for p in top_periods]
    
        # Fit multi-sine reconstruction
        recon_wave, fitted_params = build_multi_sine(t_seconds, filtered_signal, top_periods_simple)
    
        # Add recon to df
        minute_avg_df['recon'] = recon_wave
    
        # Predict next peaks (60-min horizon)
        next_peaks, forecast_values, future_minutes = predict_future_peaks(minute_avg_df['minute'], fitted_params, horizon_minutes=60, n_peaks=3)

    with st.expander("📊 Time Series Analyzer", expanded=True):
        #fig = plot_multiplier_timeseries(df)
        #st.pyplot(fig)

        
        
        # Extract fitted params
        A_fit, phi_fit, offset_fit = params
        print(f"Sine Wave Params — Amplitude: {A_fit:.2f}, Phase: {phi_fit:.2f}, Offset: {offset_fit:.2f}")

        # Generate predicted sine wave
        predicted_wave = sine_model(time, A_fit, phi_fit, offset_fit)

        
        
        # Append it to dataframe for plotting
        minute_avg_df['sine_wave'] = predicted_wave
        

        # Detect local maxima (peak timestamps)
        second_derivative = np.diff(np.sign(np.diff(predicted_wave)))
        peak_indices = np.where(second_derivative == -2)[0] + 1  # adjust for diff offset
        # Troughs: local minima where slope goes from - to +
        trough_indices = np.where(second_derivative == 2)[0] + 1
        
        # Extract peak times (minute) and corresponding sine wave values
        peak_times = minute_avg_df['minute'].iloc[peak_indices].values
        peak_values = predicted_wave[peak_indices]

        trough_times = minute_avg_df['minute'].iloc[trough_indices].values
        trough_values = predicted_wave[trough_indices]
        
        # Get next 3 upcoming peak timestamps (if available)
        
        
        next_peaks = peak_times[-3:] if len(peak_times) >= 3 else peak_times
        next_peak_values = peak_values[-3:] if len(peak_values) >= 3 else peak_values

        next_troughs = trough_times[-3:] if len(trough_times) >= 3 else trough_times
        next_trough_values = trough_values[-3:] if len(trough_values) >= 3 else trough_values

        # Predict next 3 peak times in future
        num_future_peaks = 3
        future_peaks = []
        
        # Time offset from last data point
        t_last = time[-1]
        
        # Predict next peaks from fitted sine phase
        for n in range(1, num_future_peaks + 1):
            # Solve for t when sine is at its peak: sin(ωt + φ) = 1 → ωt + φ = π/2 + 2πn
            t_peak = (np.pi / 2 + 2 * np.pi * n - phi_fit) / omega
            t_peak_abs = t_last + (t_peak % (2 * np.pi / omega))  # align it into future
            peak_minute = minute_avg_df['minute'].iloc[0] + pd.to_timedelta(int(t_peak_abs), unit='m')
            future_peaks.append(peak_minute)
        
        

        
                # Compute MACD
        # AX3: MACD over MSI
        if show_macd and 'msi_macd' in df.columns:
            fig3, ax3 = plt.subplots(figsize=(12, 2.5))
            ax3.plot(df['msi_macd'], label='MSI-MACD', color='blue')
            ax3.plot(df['msi_signal'], label='MACD Signal', color='red', linestyle='--')
            ax3.bar(df.index, df['msi_hist'], label='MACD Hist', color='gray', alpha=0.4, width=1)
            ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax3.set_ylabel('MSI-MACD')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.15)
            plot_slot = st.empty()
            with plot_slot.container():
                st.pyplot(fig3)
        
        # ---------- PLOTTING ----------
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot MACD and Signal line
        ax.plot(minute_avg_df['minute'], macd_df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(minute_avg_df['minute'], macd_df['macd_signal'], label='Signal', color='orange', linewidth=1.5)
        
        # Histogram
        #ax.bar(minute_avg_df['minute'], macd_df['macd_hist'], label='Histogram', color='gray', alpha=0.5)
        
        # Labels & legend
        ax.set_title("MACD Indicator (Minute Avg)")
        ax.set_xlabel("Minute")
        ax.set_ylabel("MACD Value")
        ax.legend()
        
        
        plt.tight_layout()
        plot_slot = st.empty()
        with plot_slot.container():
            st.pyplot(fig)
    
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(minute_avg_df['minute'], signal, label='Avg Multiplier (1-min)', alpha=0.6)
        ax.plot(minute_avg_df['minute'], predicted_wave, label='Fitted Surge Wave', color='black', linewidth=2)
       
        
        # Mark peaks
        ax.scatter(peak_times, peak_values, color='red', label='Predicted Peaks bruv', zorder=5)
        ax.scatter(trough_times, trough_values, color='purple', label='Predicted troughs bruv', zorder=5)

        ax.set_title("📈 Predictive Sine Rebuild")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_slot = st.empty()
        with plot_slot.container():
            st.pyplot(fig2)

        # --- Plot STL components ---
        st.subheader("🧪 STL Decomposition")
        
        fig3, axs = plt.subplots(figsize=(10, 4))
        
        axs.plot(minute_avg_df['minute'], signal, label='Filtered Signal', linewidth=1.8)
        
        axs.plot(minute_avg_df['minute'], trend, label='Trend', color='orange')
        
        axs.plot(minute_avg_df['minute'], seasonal, label='Seasonal', color='green')
        
        axs.plot(minute_avg_df['minute'], residual, label='Residual', color='red')
        
        axs.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plot_slot = st.empty()
        #with plot_slot.container():
        #    st.pyplot(fig3)

         # --- SETTINGS ---
        forecast_horizon = 60  # minutes ahead to project lag lines
        acf_threshold = 0.4    # for horizontal line
        
        # --- PLOTTING (upgraded) ---
        st.subheader("🔮 Combined STL + ACF + FFT Predictor")
        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        
        # ============ 1. TOP CHART: Raw filtered + Recon + Trend + Peak/Trough Lines ============
        ax[0].plot(minute_avg_df['minute'], filtered_signal, label='Filtered Signal', color='black', alpha=0.8)
        ax[0].plot(minute_avg_df['minute'], minute_avg_df['recon'], label='Reconstructed Wave', color='purple', linewidth=1.7)
        ax[0].plot(minute_avg_df['minute'], minute_avg_df['trend'], label='Trend', color='orange', linestyle='--')
        
        # Peaks & troughs from recon
        sec_deriv_recon = np.diff(np.sign(np.diff(minute_avg_df['recon'].values)))
        peak_idx = np.where(sec_deriv_recon == -2)[0] + 1
        trough_idx = np.where(sec_deriv_recon == 2)[0] + 1
        
        if len(peak_idx) > 0:
            ax[0].scatter(minute_avg_df['minute'].iloc[peak_idx],
                          minute_avg_df['recon'].iloc[peak_idx],
                          color='red', label='Recon Peaks')
            for t in minute_avg_df['minute'].iloc[peak_idx]:
                ax[0].axvline(t, color='red', linestyle=':', alpha=0.3)
        
        if len(trough_idx) > 0:
            ax[0].scatter(minute_avg_df['minute'].iloc[trough_idx],
                          minute_avg_df['recon'].iloc[trough_idx],
                          color='blue', label='Recon Troughs')
            for t in minute_avg_df['minute'].iloc[trough_idx]:
                ax[0].axvline(t, color='blue', linestyle=':', alpha=0.3)
        
        ax[0].legend(fontsize=8)
        ax[0].set_title("Signal vs Reconstructed Wave")
        
        
        plt.tight_layout()
        st.pyplot(fig)
        
      
        # 🔮 Display Wave Clock Prediction
        if len(next_peaks) > 0:
            formatted_peaks = [pd.to_datetime(p).strftime('%H:%M') for p in next_peaks]
            st.success(f"🕓 Next Surge Peaks (Wave Clock): {', '.join(formatted_peaks)}")

        if len(next_troughs) > 0:
            formatted_troughs = [pd.to_datetime(p).strftime('%H:%M') for p in next_troughs]
            st.success(f"🕓 Next troughs (Wave Clock): {', '.join(formatted_troughs)}")
        else:
            st.info("🔄 Waiting for enough data to predict wave clock...")
            
        # Display predicted peak minutes
        st.markdown("### 🔮 Next Predicted Surge Times:")
        for i, peak_time in enumerate(future_peaks, 1):
            st.write(f"Peak #{i}: {peak_time.strftime('%H:%M')}")

        
    


    with st.expander("📊 Multi-Wave Trap Scanner"):
        st.write("This shows smoothed multiplier waves across multiple timeframes.")
        peak_dict, trough_dict = multi_wave_trap_scanner(df, windows=[1, 3, 5, 10])

        

    
    FIB_WINDOWS = [3, 5, 8, 13, 21,34]  
    for w in FIB_WINDOWS:
        if len(df) < w :
            continue

        df['range_width']  = df['multiplier'].rolling(w).apply(lambda x: x.max() - x.min(), raw=True)
        #atr_series = atr_series.bfill().fillna(0)
        
   
    
    
    #st.subheader("🌀 ALIEN MWATR OSCILLATOR (Ultra-Mode)")
    
    FIB_WINDOWS = [3, 5, 8, 13, 21,34]
    
    
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
