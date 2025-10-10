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

    raw_signals = {}  # keep raw signals per window
    organic_signals = {}  # keep organic signals per window

    for w in windows:
        # Aggregate to higher minute frames
        df_w = (
            round_df.resample(f"{w}T", on="timestamp")['multiplier']
            .mean()
            .reset_index()
            .rename(columns={'multiplier': 'multiplier', 'timestamp': 'minute'})
        )

        signal, df_w = build_responsive_signal(df_w)
        raw_signals[w] = (df_w['minute'], signal)

        # Peak/trough detection
        peaks, _ = find_peaks(signal, distance=2)
        troughs, _ = find_peaks(-signal, distance=2)

        peak_dict[w] = (df_w['minute'].iloc[peaks], signal[peaks])
        trough_dict[w] = (df_w['minute'].iloc[troughs], signal[troughs])

        # Plot
        ax.plot(df_w['minute'], signal, label=f"{w}-min Wave")
        ax.scatter(df_w['minute'].iloc[peaks], signal[peaks], color='red', marker='o', s=40)
        ax.scatter(df_w['minute'].iloc[troughs], signal[troughs], color='purple', marker='x', s=40)

        # ----- ORGANIC without pinks -----
        df_org = (
            round_df[round_df['multiplier'] < 10]  # strip pinks
            .resample(f"{w}T", on="timestamp")['multiplier']
            .mean()
            .reset_index()
            .rename(columns={'multiplier': 'multiplier', 'timestamp': 'minute'})
        )

        org_signal, df_org = build_responsive_signal(df_org)
        organic_signals[w] = (df_org['minute'], org_signal)

    ax.set_title("🔮 Multi-Wave Trap Scanner (Smoothed Higher Minute Waves)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_slot = st.empty()
    with plot_slot.container():
        st.pyplot(fig)

    # === FIGURE 2: Organic Flow (ignoring pinks) ===
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for w, (minutes, signal) in organic_signals.items():
        ax2.plot(minutes, signal, label=f"{w}-min Organic Flow")
    ax2.set_title("🌱 Organic Flow (No Pinks)")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_slot = st.empty()
    with plot_slot.container():
        st.pyplot(fig2)

    # === FIGURE 3: PMF (Raw - Organic) ===
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for w in windows:
        if w in raw_signals and w in organic_signals:
            raw_minutes, raw_sig = raw_signals[w]
            org_minutes, org_sig = organic_signals[w]

            # Align indexes safely
            df_merge = pd.DataFrame({'minute': raw_minutes, 'raw': raw_sig}).merge(
                pd.DataFrame({'minute': org_minutes, 'organic': org_sig}),
                on='minute', how='inner'
            )
            df_merge['pmf'] = df_merge['raw'] - df_merge['organic']

            ax3.plot(df_merge['minute'], df_merge['pmf'], label=f"{w}-min PMF")

    ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax3.set_title("🎭 Pink Manipulation Factor (PMF)")
    ax3.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_slot = st.empty()
    with plot_slot.container():
        st.pyplot(fig3)
    

    
    return peak_dict, trough_dict
    
@st.cache_data
def compute_bsf_bmf_composite(
    df, windows=(1,3,5,10), blue_cut=1.2, z_win=120, use_type_if_available=True
):
    """
    Build per-TF Blue Suppression/Manipulation measures:
      - no_blues_mean: resampled mean with blues removed
      - BSF = raw_mean - no_blues_mean   (↑ means fewer blues; safer)
      - BMF = -BSF                       (↑ means more blues; riskier)

    Returns:
      bsf_1m: DataFrame aligned to 1-minute with per-TF BSF, BMF, composite_BSF and regime
      z_bsf:  z-scored per-TF BSF for visual comparison
      weights: dict of TF weights used in composite
    """
    df = df.dropna(subset=['timestamp']).copy().sort_values('timestamp')

    # ---- blue mask (prefer labels if available) ----
    if use_type_if_available and 'type' in df.columns:
        # Treat rows explicitly labeled "Blue" as blue; otherwise fallback to threshold.
        blue_mask = df['type'].str.lower().eq('blue')
        # If label coverage is partial, OR them with threshold to be safe:
        if blue_mask.mean() < 0.05:  # very few labeled? fallback
            blue_mask = blue_mask | (df['multiplier'] <= blue_cut)
    else:
        blue_mask = df['multiplier'] <= blue_cut

    def resample_mean(data, w):
        return (
            data.resample(f"{w}T", on="timestamp")['multiplier']
            .mean()
            .rename(f'mean_{w}')
        )

    bsf_series, bmf_series = {}, {}

    for w in windows:
        raw_w = resample_mean(df, w)
        noblues_w = resample_mean(df[~blue_mask], w)

        # align & fill
        idx = raw_w.index.union(noblues_w.index)
        raw_w = raw_w.reindex(idx).interpolate().ffill().bfill()
        noblues_w = noblues_w.reindex(idx).interpolate().ffill().bfill()

        bsf_w = (raw_w - noblues_w).rename(f'bsf_{w}')
        bmf_w = (-bsf_w).rename(f'bmf_{w}')

        bsf_series[w] = bsf_w
        bmf_series[w] = bmf_w

    # Common 1-minute grid
    start = min(s.index.min() for s in bsf_series.values())
    end   = max(s.index.max() for s in bsf_series.values())
    grid = pd.date_range(start=start, end=end, freq='T')

    bsf_1m = pd.DataFrame(index=grid)
    for w, s in bsf_series.items():
        up = s.reindex(grid).interpolate().ffill().bfill()
        tmp_df = pd.DataFrame({'minute': up.index, 'multiplier': up.values})
        smoothed, _ = build_responsive_signal(tmp_df)
        bsf_1m[f'bsf_{w}'] = smoothed

    # z-score for comparability
    roll_mean = bsf_1m.rolling(z_win, min_periods=max(10, z_win//6)).mean()
    roll_std  = bsf_1m.rolling(z_win, min_periods=max(10, z_win//6)).std().replace(0, np.nan)
    z = ((bsf_1m - roll_mean) / roll_std).fillna(0.0)
    z_bsf = z.copy()

    # weights: slow TF sets environment; mid TFs trigger; fast TF times
    default_weights = {1:0.15, 3:0.3, 5:0.3, 10:0.25}
    weights = {w: default_weights.get(w, 0.0) for w in windows}

    # down-weight long TF if short history
    if len(bsf_1m.index) < 200 and 10 in weights:
        weights[10] *= 0.6

    composite = sum(z_bsf[f'bsf_{w}'] * weights[w] for w in windows if f'bsf_{w}' in z_bsf.columns)
    bsf_1m['composite_bsf'] = composite

    slope = bsf_1m['composite_bsf'].diff()
    bsf_1m['regime'] = np.select(
        [ (composite >  0.5) & (slope > 0),
          (composite < -0.5) & (slope < 0)],
        ['Blue-Suppressed (safer to clip low)', 'Blue-Injected (stand down)'],
        default='Neutral'
    )

    # Also add per-TF BMF (optional; mirrors sign)
    for w, s in bmf_series.items():
        up = s.reindex(grid).interpolate().ffill().bfill()
        tmp_df = pd.DataFrame({'minute': up.index, 'multiplier': up.values})
        smoothed, _ = build_responsive_signal(tmp_df)
        bsf_1m[f'bmf_{w}'] = smoothed  # negative of bsf_{w} (conceptually)

    return bsf_1m, z_bsf, weights


@st.cache_data
def compute_organic_signal_and_slope_composites(
    df,
    windows=(1, 3, 5, 10),
    pink_cut=10.0,
    z_win=120,
    weights=None
):
    """
    Returns:
      organic_1m : DataFrame indexed by 1-min grid with columns:
         - org_{w}         : smoothed organic (no-pink) signal for window w
         - slope_{w}       : gradient of org_{w}
      z_signals   : DataFrame with z-scored signals per TF (z_signal_{w})
      z_slopes    : DataFrame with z-scored slopes per TF (z_slope_{w})
      composites  : dict with:
         - 'composite_signal' : weighted sum of z_signals (series)
         - 'composite_slope'  : gradient(composite_signal) or weighted z_slopes
         - 'weights'          : used weights dict
    Notes:
      - Uses your build_responsive_signal(...) which expects tmp_df with columns ['minute','multiplier'].
      - Aligns everything to a 1-minute grid across the full span of df.
    """
    import numpy as np
    import pandas as pd

    # guards
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    df = df.dropna(subset=['timestamp']).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')

    # prepare grid
    grid_start = df['timestamp'].dt.floor('min').min()
    grid_end   = df['timestamp'].dt.floor('min').max()
    grid = pd.date_range(start=grid_start, end=grid_end, freq='T')

    # filter organic rounds (no pinks)
    df_org_all = df[df['multiplier'] < pink_cut]
    if df_org_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    organic_1m = pd.DataFrame(index=grid)
    z_signals = pd.DataFrame(index=grid)
    z_slopes = pd.DataFrame(index=grid)
    present_windows = []

    for w in windows:
        try:
            org_w = df_org_all.resample(f"{w}T", on='timestamp')['multiplier'].mean()
        except Exception:
            continue
        if org_w.dropna().empty:
            continue

        # upsample to 1-min grid and fill
        up = org_w.reindex(grid).interpolate().ffill().bfill()

        # prepare tmp df for your build_responsive_signal
        tmp_df = pd.DataFrame({'minute': up.index, 'multiplier': up.values})

        # smooth (uses your sgolay inside)
        try:
            smoothed, _ = build_responsive_signal(tmp_df)
        except Exception:
            smoothed = pd.Series(up.values, index=up.index).rolling(window=max(3, min(len(up), 5)), center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill').values

        organic_1m[f'org_{w}'] = smoothed
        # slope (per-minute)
        slope = np.gradient(smoothed)
        organic_1m[f'slope_{w}'] = slope

        # z-score the SIGNAL (rolling) so TFs are comparable for composite signal
        s_series = pd.Series(smoothed, index=grid)
        s_mean = s_series.rolling(z_win, min_periods=max(10, z_win//6)).mean()
        s_std  = s_series.rolling(z_win, min_periods=max(10, z_win//6)).std().replace(0, np.nan)
        z_sig = ((s_series - s_mean) / s_std).fillna(0.0)
        z_signals[f'z_signal_{w}'] = z_sig

        # z-score the slope as well (for composite slope)
        slope_s = pd.Series(slope, index=grid)
        sm = slope_s.rolling(z_win, min_periods=max(10, z_win//6)).mean()
        ss = slope_s.rolling(z_win, min_periods=max(10, z_win//6)).std().replace(0, np.nan)
        z_slp = ((slope_s - sm) / ss).fillna(0.0)
        z_slopes[f'z_slope_{w}'] = z_slp

        present_windows.append(w)

    if z_signals.empty:
        return organic_1m, z_signals, z_slopes, {}

    # default weights (slow sets bias)
    if weights is None:
        default_weights = {1:0.15, 3:0.30, 5:0.30, 10:0.25}
        weights = {w: default_weights.get(w, 0.0) for w in windows}

    # keep only present windows and normalize
    used_weights = {w: weights.get(w, 0.0) for w in present_windows}
    s = sum(used_weights.values()) or 1.0
    used_weights = {w: (used_weights[w] / s) for w in used_weights}

    # composite signal = weighted sum of z_signals
    composite_signal = None
    for w in present_windows:
        col = f'z_signal_{w}'
        if col in z_signals.columns:
            if composite_signal is None:
                composite_signal = z_signals[col] * used_weights.get(w, 0.0)
            else:
                composite_signal = composite_signal + z_signals[col] * used_weights.get(w, 0.0)
    if composite_signal is None:
        composite_signal = pd.Series(0.0, index=grid)

    # composite slope: two options (choose both)
    # A) gradient of composite_signal (natural, reflects change in bias)
    comp_slope_from_signal = np.gradient(composite_signal.values)
    comp_slope_from_signal = pd.Series(comp_slope_from_signal, index=grid)

    # B) weighted sum of z_slopes (direct momentum composite)
    composite_slope2 = None
    for w in present_windows:
        col = f'z_slope_{w}'
        if col in z_slopes.columns:
            if composite_slope2 is None:
                composite_slope2 = z_slopes[col] * used_weights.get(w, 0.0)
            else:
                composite_slope2 = composite_slope2 + z_slopes[col] * used_weights.get(w, 0.0)
    if composite_slope2 is None:
        composite_slope2 = pd.Series(0.0, index=grid)

    # store composites and regimes
    organic_1m['composite_signal'] = composite_signal.values
    organic_1m['composite_slope'] = comp_slope_from_signal.values
    organic_1m['composite_slope_zweighted'] = composite_slope2.values

    # regime labeling (bias + momentum)
    comp = organic_1m['composite_signal']
    comp_dx = organic_1m['composite_slope']
    conds = [
        (comp > 0.6) & (comp_dx > 0),
        (comp < -0.6) & (comp_dx < 0)
    ]
    choices = ['Organic-Up (bias)', 'Organic-Down (bias)']
    organic_1m['regime'] = pd.Series(np.select(conds, choices, default='Neutral'), index=grid)

    composites = {
        'composite_signal': composite_signal,
        'composite_slope_from_signal': comp_slope_from_signal,
        'composite_slope_zweighted': composite_slope2,
        'weights': used_weights
    }

    return organic_1m, z_signals, z_slopes, composites


@st.cache_data
def compute_supertrend(df, period=10, multiplier=2.0, source="momentum"):
    """
    Supertrend calculation adapted for Momentum Tracker.
    Uses the momentum line (or other chosen column) as the source.
    """
    df = df.copy()
    src = df[source].astype(float)

    # True range approximation (adapted to momentum shifts)
    df['prev_val'] = src.shift(1)
    df['tr'] = (src - df['prev_val']).abs()
    df['atr'] = df['tr'].rolling(window=period).mean()

    # Bands
    df['upper_band'] = src - multiplier * df['atr']
    df['lower_band'] = src + multiplier * df['atr']

    # Initialize trend
    trend = [1]  # start uptrend

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        # Carry-forward logic
        upper_band = (
            max(curr['upper_band'], prev['upper_band'])
            if prev['prev_val'] > prev['upper_band']
            else curr['upper_band']
        )
        lower_band = (
            min(curr['lower_band'], prev['lower_band'])
            if prev['prev_val'] < prev['lower_band']
            else curr['lower_band']
        )

        # Switch trend if thresholds crossed
        if trend[-1] == -1 and curr['prev_val'] > lower_band:
            trend.append(1)
        elif trend[-1] == 1 and curr['prev_val'] < upper_band:
            trend.append(-1)
        else:
            trend.append(trend[-1])

        # Update bands
        df.at[df.index[i], 'upper_band'] = upper_band
        df.at[df.index[i], 'lower_band'] = lower_band

    # Final outputs
    df["trend"] = trend
    df["supertrend"] = np.where(df["trend"] == 1, df["upper_band"], df["lower_band"])
    df["buy_signal"] = (df["trend"] == 1) & (pd.Series(trend).shift(1) == -1)
    df["sell_signal"] = (df["trend"] == -1) & (pd.Series(trend).shift(1) == 1)

    return df

@st.cache_data(ttl=600, show_spinner=False)
def compute_momentum_macd(minute_avg_df, col='momentum', fast=6, slow=13, signal=5):
    df = minute_avg_df.copy()
    df['ema_fast'] = df[col].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df[col].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df



def add_stl_decomposition(df, period=12, source="momentum"):
    """
    Adds STL decomposition (trend, seasonal, residual) 
    on the raw momentum line.
    
    Args:
        df (pd.DataFrame): momentum tracker dataframe
        period (int): seasonal cycle length (in rounds, tuneable)
        source (str): which series to decompose, default='momentum'
    """
    df = df.copy()
    signal = df[source].astype(float).values

    # --- STL Decomposition ---
    stl = STL(signal, period=period, robust=True)
    res = stl.fit()

    df["stl_trend"] = res.trend
    df["stl_seasonal"] = res.seasonal
    df["stl_residual"] = res.resid

    return df


@st.cache_data(show_spinner=False)
def compute_momentum_tracker(df, alpha=0.75):
    """
    Momentum tracker with Fibonacci trap detection and tactical visuals.
    - Uses raw precision scoring on multipliers
    - Builds momentum line
    - Detects Fibonacci trap danger zones
    - Flags pink reaction zones
    - Returns tactical overlay chart (matplotlib figure)
    """

    # === 1. Precision scoring === #
    def score_round(multiplier):
        if multiplier < 1.5:
            return -1.5
        return np.interp(
            multiplier,
            [1.5, 2.0, 5.0, 10.0, 20.0],
            [-1.0, 1.0, 1.5, 2.0, 3.0]
        )

    df = df.copy()
    df['scores'] = df['multiplier'].apply(score_round)

    # === 2. Momentum line (cumulative scoring) === #
    df['momentum'] = df['scores'].cumsum()

    # === 3. Bollinger Bands on momentum === #
    df['bb_mid_10'], df['bb_upper_10'], df['bb_lower_10'] = bollinger_bands(
        df['momentum'], window=10, num_std=1.5
    )

    # === 4. Fitted sine wave cycle === #
    signal = df['momentum'].values
    N = len(signal)
    peaks, troughs = [], []
    ghost_marker = None

    if N > 8:
        signal = savgol_filter(signal, window_length=min(7, N-(N%2==0)), polyorder=2)
        T = 1.0
        time = np.arange(N)

        # FFT
        yf = rfft(signal)
        xf = rfftfreq(N, T)[:N // 2]
        fft_magnitude = 2.0 / N * np.abs(yf[0:N // 2])

        if len(fft_magnitude[1:]) > 0:
            dominant_index = np.argmax(fft_magnitude[1:]) + 1
            dominant_freq = xf[dominant_index]
            omega = 2 * np.pi * dominant_freq

            def sine_model(t, A, phi, offset):
                return A * np.sin(omega * t + phi) + offset

            params, _ = curve_fit(sine_model, time, signal, p0=[1, 0, np.mean(signal)])
            A_fit, phi_fit, offset_fit = params
            df['sine_wave'] = sine_model(time, A_fit, phi_fit, offset_fit)

            # Extrema detection
            second_derivative = np.diff(np.sign(np.diff(df['sine_wave'])))
            peaks = list(np.where(second_derivative == -2)[0] + 1)
            troughs = list(np.where(second_derivative == 2)[0] + 1)

            # Estimate cycle period
            cycle_lengths = []
            if len(peaks) >= 2:
                cycle_lengths.append(peaks[-1] - peaks[-2])
            if len(troughs) >= 2:
                cycle_lengths.append(troughs[-1] - troughs[-2])

            if cycle_lengths:
                est_period = int(np.mean(cycle_lengths))

                # Project next extrema based on last one
                if peaks and (not troughs or peaks[-1] > troughs[-1]):
                    next_idx = peaks[-1] + est_period
                    if next_idx < N + est_period:  # allow projection just beyond data
                        ghost_marker = {
                            "type": "peak",
                            "x": next_idx,
                            "y": sine_model(next_idx, A_fit, phi_fit, offset_fit)
                        }
                elif troughs:
                    next_idx = troughs[-1] + est_period
                    if next_idx < N + est_period:
                        ghost_marker = {
                            "type": "trough",
                            "x": next_idx,
                            "y": sine_model(next_idx, A_fit, phi_fit, offset_fit)
                        }



    # === 3. Fibonacci danger zones (trap detection) === #
    danger_zones = [
        i for i in range(4, len(df))
        if sum(df['multiplier'].iloc[i-4:i+1] < 2.0) >= 4
    ]

    # === 4. Pink reaction zones === #
    pink_mask = df['multiplier'] >= 10
    pink_zones = {
        'indices': df.index[pink_mask].tolist(),
        'multipliers': df['multiplier'][pink_mask].tolist()
    }

    # === Supertrend calculation ===
    df = compute_supertrend(df, period=10, multiplier=2.0, source="momentum")
    
    # === STL decomposition ===
    df = add_stl_decomposition(df, period=12, source="momentum")
    
    


    # === 5. Tactical overlay chart === #
    #plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Momentum line
    momentum_smooth = df['momentum'].ewm(alpha=alpha).mean()
    ax.plot(momentum_smooth, color='#00fffa', lw=2,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgecolor='white', zorder=4, label="Momentum")

    # Bollinger Bands
    ax.plot(df['bb_mid_10'], color='yellow', lw=1.2, alpha=0.6, label="BB Mid (10)")
    ax.plot(df['bb_upper_10'], color='red', lw=1, alpha=0.4, linestyle="--", label="BB Upper")
    ax.plot(df['bb_lower_10'], color='green', lw=1, alpha=0.4, linestyle="--", label="BB Lower")
    ax.fill_between(df.index, df['bb_lower_10'], df['bb_upper_10'],
                    color='gray', alpha=0.1)


    # Supertrend line
    ax.plot(df['supertrend'], color='orange', lw=1.8, label="Supertrend")
    
    # Buy/Sell markers
    ax.scatter(df.index[df['buy_signal']], df['supertrend'][df['buy_signal']],
               marker='^', color='lime', edgecolor='black', s=90, zorder=8, label="Buy Signal")
    
    ax.scatter(df.index[df['sell_signal']], df['supertrend'][df['sell_signal']],
               marker='v', color='red', edgecolor='black', s=90, zorder=8, label="Sell Signal")

    # Plot STL trend as a thicker background line
    #ax.plot(df['stl_trend'], color='yellow', lw=2.2, alpha=0.8, label="STL Trend")
    
    # Seasonal component as dotted wave (hidden cycles)
    ax.plot(df['stl_seasonal'], color='navy', lw=1.5, linestyle='--', alpha=0.7, label="STL Seasonal")
    
    # Optional: residual as thin gray line for noise view
    #ax.plot(df['stl_residual'], color='gray', lw=0.8, alpha=0.5, label="STL Residual")

     # Fitted sine wave + extrema
    if 'sine_wave' in df.columns:
        ax.plot(df['sine_wave'], color='black', lw=2, label="Fitted Cycle")

        ax.scatter(peaks, df['sine_wave'].iloc[peaks],
                   color='red', edgecolor='white', s=80, zorder=6, label="Cycle Peaks")
        ax.scatter(troughs, df['sine_wave'].iloc[troughs],
                   color='lime', edgecolor='white', s=80, zorder=6, label="Cycle Troughs")

        # Ghost marker
        if ghost_marker:
            ax.scatter(
                ghost_marker["x"], ghost_marker["y"],
                facecolors='none',
                edgecolors='red' if ghost_marker["type"] == "peak" else 'lime',
                s=120, lw=2, alpha=0.7, zorder=7,
                label=f"Projected {ghost_marker['type'].capitalize()}"
            )

    # Pink zones
    for mult, idx in zip(pink_zones['multipliers'], pink_zones['indices']):
        #if idx < len(df['momentum']):
        pink_level = df['momentum'].iloc[idx]
        ax.axhline(y=pink_level, color='purple', linestyle='--',
                    linewidth=1.2, alpha=0.6)
        ax.scatter(idx, pink_level, color='#ff00ff',
                    edgecolor='black', s=60, zorder=5)
        ax.axvline(x=idx, color='purple', linestyle=':', alpha=0.4)

    # Danger zones (red spans)
    for zone in danger_zones:
        ax.axvspan(zone - 0.5, zone + 0.5, color='#d50000', alpha=0.15)

    ax.set_title("CYA TACTICAL OVERLAY v6.2 (BB + Fitted Cycle)",
                 color='#00fffa', fontsize=18, weight='bold')
    #ax.set_facecolor('#000000')
    ax.legend(loc="upper left")

    #plt.tight_layout()

   # === 8. Store results === #
    st.session_state.momentum_line = df['momentum'].tolist()
    st.session_state.danger_zones = danger_zones
    st.session_state.pink_zones = pink_zones
    st.session_state.cycle_peaks = peaks
    st.session_state.cycle_troughs = troughs
    st.session_state.ghost_marker = ghost_marker

    
    return df, fig


@st.cache_data(show_spinner=False)
def compute_momentum_time_series(df):
    """
    Momentum Time Series Analyzer:
    - Averages momentum scores per minute
    - Applies SavGol smoothing
    - Fits FFT-derived sine curve
    - Detects peaks/troughs
    - Projects forward next cycle turns
    """

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')

    # === Average momentum score per minute ===
    minute_avg_df = df.groupby('minute').agg({'momentum': 'mean'}).reset_index()

    # Fill gaps for clean FFT
    minute_avg_df.set_index('minute', inplace=True)
    minute_avg_df = minute_avg_df.resample('1min').mean().interpolate()
    minute_avg_df.reset_index(inplace=True)

    signal = minute_avg_df['momentum'].values
    N = len(signal)

    if N < 6:
        st.warning(f"⚠️ Low data length for FFT ({N} points). Accuracy may be poor.")

    # Smooth signal for cleaner cycle detection
    from scipy.signal import savgol_filter
    if N >= 5:
        signal = savgol_filter(signal, window_length=min(11, N-(N%2==0)), polyorder=2)

    # FFT prep
    from scipy.fft import rfft, rfftfreq
    from scipy.optimize import curve_fit

    T = 60.0  # 1 min = 60s
    time = np.arange(N)
    yf = rfft(signal)
    xf = rfftfreq(N, T)[:N // 2]

    fft_magnitude = 2.0 / N * np.abs(yf[0:N // 2])
    if len(fft_magnitude[1:]) == 0:
        raise ValueError("🚫 Empty FFT magnitude — not enough signal.")

    # Find dominant cycle
    dominant_index = np.argmax(fft_magnitude[1:]) + 1
    dominant_freq = xf[dominant_index]
    omega = 2 * np.pi * dominant_freq

    def sine_model(t, A, phi, offset):
        return A * np.sin(omega * t + phi) + offset

    params, _ = curve_fit(sine_model, time, signal, p0=[1, 0, np.mean(signal)])
    A_fit, phi_fit, offset_fit = params
    predicted_wave = sine_model(time, A_fit, phi_fit, offset_fit)

    # Add fitted curve
    minute_avg_df['sine_wave'] = predicted_wave

    # Detect extrema
    second_derivative = np.diff(np.sign(np.diff(predicted_wave)))
    peak_indices = np.where(second_derivative == -2)[0] + 1
    trough_indices = np.where(second_derivative == 2)[0] + 1

    peak_times = minute_avg_df['minute'].iloc[peak_indices].values
    peak_values = predicted_wave[peak_indices]
    trough_times = minute_avg_df['minute'].iloc[trough_indices].values
    trough_values = predicted_wave[trough_indices]

    # Project next expected peak/trough
    cycle_length = int(round(2 * np.pi / omega))
    next_peak_time = minute_avg_df['minute'].iloc[-1] + pd.to_timedelta(cycle_length // 2, unit='m')
    next_trough_time = minute_avg_df['minute'].iloc[-1] + pd.to_timedelta(cycle_length, unit='m')

    # --- Plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(minute_avg_df['minute'], signal, label='Avg Momentum (1-min)', alpha=0.6)
    ax2.plot(minute_avg_df['minute'], predicted_wave, label='Fitted Cycle', color='black', linewidth=2)

    ax2.scatter(peak_times, peak_values, color='red', marker='o', s=60, label='Peaks')
    ax2.scatter(trough_times, trough_values, color='purple', marker='o', s=60, label='Troughs')

    # Ghost markers for projected next cycle
    #ax2.scatter(next_peak_time, predicted_wave[-1], color='red', marker='o', s=80, alpha=0.3, label='Next Peak (Projected)')
    #ax2.scatter(next_trough_time, predicted_wave[-1], color='purple', marker='o', s=80, alpha=0.3, label='Next Trough (Projected)')

    ax2.set_title("⏳ Momentum Time Series Analyzer (MTSA)", fontsize=14, color='cyan')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return minute_avg_df, fig2


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
    #df["momentum"] = df["score"].cumsum()
    df["round_index"] = range(len(df))
    # Define latest_msi safely
    latest_msi = df["msi"].iloc[-1] if not df["msi"].isna().all() else 0
    latest_tpi = compute_tpi(df, window=window_size)
    
    
    #df["bb_mid"]   = df["msi"].rolling(WINDOW_SIZE).mean()
    #df["bb_std"]   = df["msi"].rolling(WINDOW_SIZE).std()
    #df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    #df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    #df["bandwidth"] = df["bb_upper"] - df["bb_lower"]
    
    # === Detect Squeeze Zones (Low Volatility)
    #squeeze_threshold = df["bandwidth"].rolling(10).quantile(0.25)
    #df["squeeze_flag"] = df["bandwidth"] < squeeze_threshold
    
    # === Directional Breakout Detector
    #df["breakout_up"]   = df["msi"] > df["bb_upper"]
    #df["breakout_down"] = df["msi"] < df["bb_lower"]
    
    # === Slope & Acceleration
    #df["msi_slope"]  = df["msi"].diff()
    #df["msi_accel"]  = df["msi_slope"].diff()

    # MSI CALCULATION (Momentum Score Index)
    window_size = min(window_size, len(df))
    recent_df = df.tail(window_size)
    msi_score = recent_df['score'].mean() if not recent_df.empty else 0
    msi_color = 'green' if msi_score > 0.5 else ('yellow' if msi_score > 0 else 'red')

    #df = enhanced_msi_analysis(df)
    #df = compute_momentum_adaptive_ma(df)
    #df = compute_msi_macd(df, msi_col='msi')

    # Multi-window BBs on MSI
    #df["bb_mid_20"], df["bb_upper_20"], df["bb_lower_20"] = bollinger_bands(df["msi"], 20, 2)
    #df["bb_mid_10"], df["bb_upper_10"], df["bb_lower_10"] = bollinger_bands(df["msi"], 10, 1.5)
    #df["bb_mid_40"], df["bb_upper_40"], df["bb_lower_40"] = bollinger_bands(df["msi"], 40, 2.5)
    #df['bandwidth'] = df["bb_upper_10"] - df["bb_lower_10"]  # Width of the band
    
    # Compute slope (1st derivative) for upper/lower bands
    #df['upper_slope'] = df["bb_upper_10"].diff()
    #df['lower_slope'] = df["bb_lower_10"].diff()
    
    # Compute acceleration (2nd derivative) for upper/lower bands
    #df['upper_accel'] = df['upper_slope'].diff()
    #df['lower_accel'] = df['lower_slope'].diff()
    
    # How fast the band is expanding or shrinking
    #df['bandwidth_delta'] = df['bandwidth'].diff()
    
    # Pull latest values from the last row
    latest = df.iloc[-1] if not df.empty else pd.Series()

    

    # === Ichimoku Cloud on MSI ===
    #high_9  = df["msi"].rolling(window=9).max()
    #low_9   = df["msi"].rolling(window=9).min()
    #df["tenkan"] = (high_9 + low_9) / 2
    
    #high_26 = df["msi"].rolling(window=26).max()
    #low_26  = df["msi"].rolling(window=26).min()
    #df["kijun"] = (high_26 + low_26) / 2

    #high_3 = df["msi"].rolling(3).max()
    #low_3 = df["msi"].rolling(3).min()
    #df["mini_tenkan"] = (high_3 + low_3)/2

    #high_5 = df["msi"].rolling(5).max()
    #low_5 = df["msi"].rolling(5).min()
    #df["mini_kijun"] = (high_5 + low_5)/2

    #high_2 = df["msi"].rolling(1).max()
    #low_2 = df["msi"].rolling(1).min()
    #df["nano_tenkan"] = df["msi"].ewm(span=2).mean()

    # Projected Senkou A — mini average of short-term structure
    #df["mini_senkou_a"] = ((df["mini_tenkan"] + df["mini_kijun"]) / 2).shift(6)
    
    # Projected Senkou B — mini-range memory, 12-period HL midpoint
    #high_12 = df["msi"].rolling(12).max()
    #low_12 = df["msi"].rolling(12).min()
    #df["mini_senkou_b"] = ((high_12 + low_12) / 2).shift(6)

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
    
    #df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    
    #high_52 = df["msi"].rolling(window=52).max()
    #low_52  = df["msi"].rolling(window=52).min()
    #df["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
    
    #df["chikou"] = df["msi"].shift(-26)
    #df = compute_supertrend(df, period=10, multiplier=2.0, source="msi")

    # Custom Stochastic Mini-Momentum Index (SMMI)
    #lowest = df["momentum_impulse"].rolling(5).min()
    #highest = df["momentum_impulse"].rolling(5).max()
    #df["smmi"] = 100 * ((df["momentum_impulse"] - lowest) / (highest - lowest))


    # Core Fibonacci multipliers
    #fib_ratios = [1.0, 1.618, 2.618]
    
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
    
        
          
    
    # Initialize variables
    upper_slope = (0, )
    lower_slope = (0, )
    upper_accel = (0, )
    lower_accel = (0, )
    bandwidth = (0, )
    bandwidth_delta = (0, )
        
    
    
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
        [1.5, 10, 20]
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
    
    

    #spiral_detector = NaturalFibonacciSpiralDetector(df, window_size=selected_window)
    #spiral_centers = spiral_detector.detect_spirals()

    #spiral_echoes = get_spiral_echoes(spiral_centers, df)
    # Assuming df is your main DataFrame
    max_rounds = len(df)
    
    #true_flp_watchlist = project_true_forward_flp(spiral_centers, fib_layers=selected_fib_layers, max_rounds=max_rounds)
    #recent_scores = df['multiplier'].tail(34)  # use biggest fib window
    #current_msi_values= [df[f"msi_{w}"].iloc[-1] for w in selected_msi_windows]
    #current_slopes= [df[f"slope_{w}"].iloc[-1] for w in selected_msi_windows]
    #slope_history_series = [df[f"slope_{w}"].tail(5).tolist() for w in selected_msi_windows]

    
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
    
    

    # Ensure timestamp is parsed
    #df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    #df = df.dropna(subset=['timestamp'])
    
    # Round to nearest second (for consistent time axis)
    #df['second'] = df['timestamp'].dt.floor('s')
    
    # Group by each second → average multiplier
    #sec_df = df.groupby('second').agg({'multiplier': 'mean'}).reset_index()
    
    # Fill missing seconds (important for smooth EWM spans)
    #sec_df.set_index('second', inplace=True)
    #sec_df = sec_df.resample('1S').mean().interpolate()
    #sec_df.reset_index(inplace=True)
    
    # Build clean second-level series
    #sec_series = sec_df.set_index('second')['multiplier']
    
    # Fibonacci EWM lines in *seconds*
    #fib_spans = [34, 55, 91]
    #fib_df = pd.DataFrame({'time': sec_series.index, 'multiplier_sec': sec_series})
    #for s in fib_spans:
        #fib_df[f'fib{s}'] = sec_series.ewm(span=s, adjust=False).mean().values
        #fib_df[f'fib{s}']= savgol_filter(fib_df[f'fib{s}'], window_length=5 if N >= 5 else N, polyorder=2)
    
  
    
    # ========== PLOT ==========
    #with st.expander("⏱️ Fibonacci Time Map (34s / 55s / 91s)", expanded=False):
        #fig_fib, ax_fib = plt.subplots(figsize=(12, 4))
        #ax_fib.plot(fib_df['time'], fib_df['multiplier_sec'], label='Multiplier (1s)', alpha=0.35)
        #ax_fib.plot(fib_df['time'], fib_df['fib34'], label='fib34', linewidth=1.2)
        #ax_fib.plot(fib_df['time'], fib_df['fib55'], label='fib55', linewidth=1.2)
        #ax_fib.plot(fib_df['time'], fib_df['fib91'], label='fib91', linewidth=1.2)
        #ax_fib.set_title("Fibonacci-Timed Signal Lines (Second-Level)")
        #ax_fib.legend(loc='upper left')
        #plt.tight_layout()
        #st.pyplot(fig_fib)
    
        



    df, battle_fig = compute_momentum_tracker(df)

    with st.expander("⚔️ Tactical Overlay", expanded=True):
        st.pyplot(battle_fig)

    
        minute_avg_df, fig2 = compute_momentum_time_series(df)
        
        # Compute MACD on the 1-min averaged momentum
        macd_df = compute_momentum_macd(minute_avg_df, col='momentum')

        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 3), sharex=False)

        ax3.plot(macd_df['minute'], macd_df['macd'], label='MACD', color='cyan', linewidth=1.5)
        ax3.plot(macd_df['minute'], macd_df['macd_signal'], label='Signal Line', color='orange', linewidth=1)
        ax3.bar(macd_df['minute'], macd_df['macd_hist'], color=np.where(macd_df['macd_hist']>0, 'green', 'red'),
                alpha=0.4, label='Histogram')
        
        ax3.axhline(0, color='white', linewidth=0.8, alpha=0.5)
        ax3.set_title("🔍 MACD on Momentum (MTSA Signal)", color='cyan', fontsize=13)
        ax3.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig3)

    
           
    #with st.expander("📊 Multi-Wave Trap Scanner", expanded=True):
        #st.write("This shows smoothed multiplier waves across multiple timeframes.")
        #peak_dict, trough_dict = multi_wave_trap_scanner(df, windows=[1, 3, 5, 10])

    


    
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
