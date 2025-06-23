import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal import hilbert
import math
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import gridspec
import time

# ======================= CONFIG ==========================
st.set_page_config(page_title="Aviator Crash Predictor", layout="wide")
st.title("🎯 Aviator Crash Predictor: TDI+THRE Enhanced")

# ================ SESSION STATE INIT =====================
if "rounds" not in st.session_state:
    st.session_state.rounds = []
if "forecast_msi" not in st.session_state:
    st.session_state.forecast_msi = []
if "current_mult" not in st.session_state:
    st.session_state.current_mult = 2.0

# ================ CONFIGURATION SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Configuration Parameters")
    WINDOW_SIZE = st.slider("RSI Window Size", 5, 100, 14)
    PINK_THRESHOLD = st.number_input("Pink Threshold", value=10.0)
    SIGNAL_PERIOD = st.slider("Signal Line Period", 3, 20, 9)
    THRE_SENSITIVITY = st.slider("THRE Sensitivity", 0.1, 2.0, 1.0, 0.1)
    
    st.header("📉 Indicator Visibility")
    show_tdi_arrows = st.checkbox("🎯 Show TDI+THRE Arrows", value=True)
    show_supertrend = st.checkbox("🟢 Show SuperTrend", value=True)
    show_ichimoku = st.checkbox("☁️ Show Ichimoku", value=True)
    
    st.header("📊 Panel Toggles")
    FAST_ENTRY_MODE = st.checkbox("⚡ Fast Entry Mode", value=True)
    show_thre = st.checkbox("🌀 THRE Panel", value=True)
    show_details = st.checkbox("🔍 Show Detailed Metrics", value=False)
    
    if st.button("🔄 Full Reset", help="Clear all historical data"):
        st.session_state.rounds = []
        st.rerun()


# =================== CORE FUNCTIONS ========================
def compute_rsi(series, period=14):
    """Compute RSI with optimized performance"""
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-9)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_thre(scores, sensitivity=1.0):
    """Compute THRE (True Harmonic Resonance Engine) with optimized performance"""
    if len(scores) < 10:
        return np.zeros(len(scores)), np.zeros(len(scores))
    
    # Apply FFT to get frequency components
    yf = rfft(scores - np.mean(scores))
    xf = rfftfreq(len(scores), 1)
    mask = (xf > 0) & (xf < 0.5)  # Focus on meaningful frequencies
    
    if not any(mask):
        return np.zeros(len(scores)), np.zeros(len(scores))
    
    # Extract relevant frequencies and amplitudes
    freqs = xf[mask]
    amps = np.abs(yf[mask]) * sensitivity
    phases = np.angle(yf[mask])
    
    # Generate harmonic matrix
    harmonic_matrix = np.zeros((len(scores), len(freqs)))
    for i, (f, p) in enumerate(zip(freqs, phases)):
        harmonic_matrix[:, i] = np.sin(2 * np.pi * f * np.arange(len(scores)) + p)
    
    # Create composite signal and normalize
    if amps.size > 0:
        composite_signal = (harmonic_matrix * amps).sum(axis=1)
        std = np.std(composite_signal)
        if std > 0:
            normalized_signal = (composite_signal - np.mean(composite_signal)) / std
            smooth_signal = pd.Series(normalized_signal).rolling(3, min_periods=1).mean()
            signal_slope = np.gradient(smooth_signal)
            return smooth_signal.values, signal_slope
    
    return np.zeros(len(scores)), np.zeros(len(scores))

def compute_bollinger_bands(series, window, num_std=2):
    """Compute Bollinger Bands with optimized performance"""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def generate_tdi_thre_signals(df):
    """Generate TDI+THRE fusion signals"""
    if len(df) < 15:
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize signal columns
    df['entry_signal'] = False
    df['exit_signal'] = False
    df['bounce_signal'] = False
    
    # Generate entry signals (RSI cross + THRE support + slope confirmation)
    df['entry_signal'] = ((df['rsi'] > df['rsi_signal']) &  # RSI crosses above signal
                         (df['rsi'].shift(1) <= df['rsi_signal'].shift(1)) &  # Confirmed cross
                         (df['thre_value'] > 0) &  # THRE support
                         (df['thre_slope'] > 0))  # Positive slope
    
    # Generate exit signals (RSI rejection + THRE collapse)
    df['exit_signal'] = ((df['rsi'] < df['rsi_signal']) &  # RSI crosses below signal
                        (df['rsi'].shift(1) >= df['rsi_signal'].shift(1)) &  # Confirmed cross
                        ((df['thre_value'] < 0) | (df['thre_slope'] < 0)))  # THRE negative or falling
    
    # Generate bounce signals (RSI bounce from band edge + THRE inflection)
    df['bounce_signal'] = ((df['rsi'] < df['rsi_lower']) &  # RSI below lower band
                          (df['rsi'].shift(1) < df['rsi_lower'].shift(1)) &  # Was below
                          (df['rsi'] > df['rsi'].shift(1)) &  # Starting to rise
                          (df['thre_slope'] > df['thre_slope'].shift(1)))  # THRE slope improving
    
    # Filter signals based on volatility conditions
    volatility = df['rsi_upper'] - df['rsi_lower']
    avg_volatility = volatility.rolling(10).mean()
    
    # In high volatility, make signals more selective
    high_vol_mask = volatility > avg_volatility * 1.5
    df.loc[high_vol_mask, 'entry_signal'] = df.loc[high_vol_mask, 'entry_signal'] & (df.loc[high_vol_mask, 'thre_value'] > 0.8)
    
    # In low volatility, make bounce signals more sensitive
    low_vol_mask = volatility < avg_volatility * 0.5
    enhanced_bounce = ((df['rsi'] < 40) & 
                      (df['rsi'] > df['rsi'].shift(1)) & 
                      (df['thre_slope'] > 0))
    df.loc[low_vol_mask, 'bounce_signal'] = df.loc[low_vol_mask, 'bounce_signal'] | enhanced_bounce[low_vol_mask]
    
    # Calculate signal strength based on THRE value and slope
    df['signal_strength'] = np.abs(df['thre_value']) * (np.abs(df['thre_slope']) + 0.1)
    
    return df


def compute_supertrend(df, period=10, multiplier=2.0):
    """Compute SuperTrend indicator"""
    df = df.copy()
    
    # Use MSI as price substitute if available, otherwise use 'score'
    src = df['msi'] if 'msi' in df.columns else df['score']
    
    # Calculate ATR approximation
    df['prev_close'] = src.shift(1)
    df['tr'] = abs(src - df['prev_close'])
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Calculate bands
    df['upper_band_st'] = src - (multiplier * df['atr'])
    df['lower_band_st'] = src + (multiplier * df['atr'])
    
    # Initialize trend
    trend = [1]  # Start with uptrend
    
    # Calculate SuperTrend
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        upper_band = max(curr['upper_band_st'], prev['upper_band_st']) if prev['prev_close'] > prev['upper_band_st'] else curr['upper_band_st']
        lower_band = min(curr['lower_band_st'], prev['lower_band_st']) if prev['prev_close'] < prev['lower_band_st'] else curr['lower_band_st']
        
        if trend[-1] == -1 and curr['prev_close'] > lower_band:
            trend.append(1)
        elif trend[-1] == 1 and curr['prev_close'] < upper_band:
            trend.append(-1)
        else:
            trend.append(trend[-1])
        
        df.at[df.index[i], 'upper_band_st'] = upper_band
        df.at[df.index[i], 'lower_band_st'] = lower_band
    
    df["trend"] = trend
    df["supertrend"] = np.where(df["trend"] == 1, df["upper_band_st"], df["lower_band_st"])
    df["buy_signal"] = (df["trend"] == 1) & (pd.Series(trend).shift(1) == -1)
    df["sell_signal"] = (df["trend"] == -1) & (pd.Series(trend).shift(1) == 1)
    
    return df

def compute_ichimoku(df):
    """Compute Ichimoku Cloud indicator"""
    df = df.copy()
    
    # Use MSI as price substitute
    src = df['msi'] if 'msi' in df.columns else df['score']
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = src.rolling(window=9).max()
    low_9 = src.rolling(window=9).min()
    df["tenkan"] = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = src.rolling(window=26).max()
    low_26 = src.rolling(window=26).min()
    df["kijun"] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 (Projected 26 periods forward)
    df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 (Projected 26 periods forward)
    high_52 = src.rolling(window=52).max()
    low_52 = src.rolling(window=52).min()
    df["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Current closing price projected 26 periods backward
    df["chikou"] = src.shift(-26)
    
    return df

def analyze_data(data, pink_threshold=10.0, window_size=14, signal_period=9, thre_sensitivity=1.0):
    """Process data and compute all indicators"""
    if data.empty:
        return None, 0
    
    start_time = time.time()
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure timestamp column is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate type based on multiplier
    df["type"] = df["multiplier"].apply(lambda x: "Pink" if x >= pink_threshold else ("Purple" if x >= 2 else "Blue"))
    
    # Calculate score for MSI
    df["score"] = df["multiplier"].apply(lambda x: 2 if x >= pink_threshold else (1 if x >= 2 else -1))
    
    # Calculate MSI (Momentum Score Index)
    df["msi"] = df["score"].rolling(window_size).sum()
    df["momentum"] = df["score"].cumsum()
    
    # Calculate Bollinger Bands on MSI
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["msi"], window_size)
    
    # Calculate RSI
    df["rsi"] = compute_rsi(df["score"], period=window_size)
    
    # Calculate RSI Bands and Signal
    df["rsi_mid"] = df["rsi"].rolling(window_size).mean()
    df["rsi_std"] = df["rsi"].rolling(window_size).std()
    df["rsi_upper"] = df["rsi_mid"] + 1.2 * df["rsi_std"]
    df["rsi_lower"] = df["rsi_mid"] - 1.2 * df["rsi_std"]
    df["rsi_signal"] = df["rsi"].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate THRE values
    thre_value, thre_slope = compute_thre(df["score"].values, sensitivity=thre_sensitivity)
    df["thre_value"] = thre_value
    df["thre_slope"] = thre_slope
    
    # Generate TDI+THRE signals
    df = generate_tdi_thre_signals(df)
    
    # Calculate SuperTrend
    df = compute_supertrend(df)
    
    # Calculate Ichimoku Cloud
    df = compute_ichimoku(df)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return df, processing_time


# =================== UI COMPONENTS ========================
def fast_entry_mode_ui(pink_threshold):
    """Fast entry mode UI component"""
    st.markdown("### ⚡ FAST ENTRY MODE")
    st.markdown("Quick enter rounds for rapid decision making")
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("➕ Blue (1.5x)", use_container_width=True):
            st.session_state.rounds.append({
                "timestamp": datetime.now(),
                "multiplier": 1.5,
                "score": -1
            })
            st.rerun()
    
    with cols[1]:
        if st.button("➕ Purple (2x)", use_container_width=True):
            st.session_state.rounds.append({
                "timestamp": datetime.now(),
                "multiplier": 2.0,
                "score": 1
            })
            st.rerun()
    
    with cols[2]:
        if st.button(f"➕ Pink ({pink_threshold}x)", use_container_width=True):
            st.session_state.rounds.append({
                "timestamp": datetime.now(),
                "multiplier": pink_threshold,
                "score": 2
            })
            st.rerun()

def thre_panel(df, thre_sensitivity):
    """THRE panel component"""
    st.subheader("🔬 True Harmonic Resonance Engine (THRE)")
    if len(df) < 10: 
        st.warning("Need at least 10 rounds to compute THRE.")
        return
    
    # Extract THRE values and slopes
    thre_value = df["thre_value"].values
    thre_slope = df["thre_slope"].values
    
    # Create a two-panel plot for THRE
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot THRE Value
    ax[0].plot(df["timestamp"], thre_value, label="THRE Resonance", color='cyan')
    ax[0].axhline(1.5, linestyle='--', color='green', alpha=0.5)
    ax[0].axhline(0.5, linestyle='--', color='blue', alpha=0.3)
    ax[0].axhline(-0.5, linestyle='--', color='orange', alpha=0.3)
    ax[0].axhline(-1.5, linestyle='--', color='red', alpha=0.5)
    ax[0].set_title("Composite Harmonic Resonance Strength")
    ax[0].legend()
    
    # Plot THRE Slope
    ax[1].plot(df["timestamp"], thre_slope, label="Δ Resonance Slope", color='purple')
    ax[1].axhline(0, linestyle=':', color='gray')
    ax[1].set_title("THRE Inflection Detector")
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Get latest values
    latest_value = thre_value[-1] if len(thre_value) > 0 else 0
    latest_slope = thre_slope[-1] if len(thre_slope) > 0 else 0
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Resonance Strength", f"{latest_value:.3f}")
    with col2:
        st.metric("📉 Δ Slope", f"{latest_slope:.3f}")
    
    # Interpretation
    if latest_value > 1.5:
        st.success("💥 High Constructive Stack — Pink Burst Risk ↑")
    elif latest_value > 0.5:
        st.info("🟣 Purple Zone — Harmonically Supported")
    elif latest_value < -1.5:
        st.error("🌪️ Collapse Zone — Blue Train Likely")
    elif latest_value < -0.5:
        st.warning("⚠️ Destructive Micro-Waves — High Risk")
    else:
        st.info("⚖️ Neutral Zone — Mid-Range Expected")

def plot_tdi_thre(df):
    """Plot TDI with THRE-powered arrows"""
    if len(df) < 5:
        st.warning("Need at least 5 rounds to plot TDI.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot RSI and Signal Line
    ax.plot(df["timestamp"], df["rsi"], label="RSI", color='black', linewidth=1.5)
    ax.plot(df["timestamp"], df["rsi_signal"], label="Signal Line", color='orange', linestyle='--')
    
    # Plot RSI Bands
    ax.plot(df["timestamp"], df["rsi_upper"], color='green', linestyle='--', alpha=0.5, label="RSI Upper Band")
    ax.plot(df["timestamp"], df["rsi_lower"], color='red', linestyle='--', alpha=0.5, label="RSI Lower Band")
    ax.fill_between(df["timestamp"], df["rsi_lower"], df["rsi_upper"], color='purple', alpha=0.1)
    
    # Plot reference lines
    ax.axhline(50, color='black', linestyle=':')
    ax.axhline(70, color='green', linestyle=':')
    ax.axhline(30, color='red', linestyle=':')
    
    # Plot THRE-powered arrows if enabled
    if show_tdi_arrows:
        # Entry signals (green triangles)
        entry_points = df[df['entry_signal']]
        for idx, row in entry_points.iterrows():
            # Size arrow based on signal strength
            signal_size = min(150, max(50, 75 * abs(row['signal_strength'])))
            ax.scatter(row['timestamp'], row['rsi'], marker='^', s=signal_size, 
                      color='lime', edgecolor='darkgreen', linewidth=1, 
                      alpha=0.8, zorder=5, label='_nolegend_')
        
        # Exit signals (red triangles)
        exit_points = df[df['exit_signal']]
        for idx, row in exit_points.iterrows():
            signal_size = min(150, max(50, 75 * abs(row['signal_strength'])))
            ax.scatter(row['timestamp'], row['rsi'], marker='v', s=signal_size, 
                      color='red', edgecolor='darkred', linewidth=1, 
                      alpha=0.8, zorder=5, label='_nolegend_')
        
        # Bounce signals (blue diamonds)
        bounce_points = df[df['bounce_signal']]
        for idx, row in bounce_points.iterrows():
            signal_size = min(150, max(50, 75 * abs(row['signal_strength'])))
            ax.scatter(row['timestamp'], row['rsi'], marker='D', s=signal_size, 
                      color='cyan', edgecolor='blue', linewidth=1, 
                      alpha=0.8, zorder=5, label='_nolegend_')
    
    # Add legend with custom markers
    if show_tdi_arrows and not df[['entry_signal', 'exit_signal', 'bounce_signal']].empty.all().all():
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', markersize=10),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan', markersize=10)
        ]
        custom_labels = ['Entry Signal', 'Exit Signal', 'Reversal/Bounce']
        
        # Add all legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + custom_lines, labels + custom_labels)
    else:
        ax.legend()
    
    ax.set_title("🧠 TDI+THRE Fusion Panel")
    ax.set_xlabel("Time")
    ax.set_ylabel("RSI Value")
    
    st.pyplot(fig)


def plot_msi_chart(df):
    """Plot MSI chart with indicators"""
    if len(df) < 2:
        st.warning("Need at least 2 rounds to plot MSI chart.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot MSI line
    ax.plot(df["timestamp"], df["msi"], label="MSI", color='black')
    
    # Plot Bollinger Bands
    ax.plot(df["timestamp"], df["bb_upper"], linestyle='--', color='green')
    ax.plot(df["timestamp"], df["bb_lower"], linestyle='--', color='red')
    ax.fill_between(df["timestamp"], df["bb_lower"], df["bb_upper"], color='gray', alpha=0.1)
    
    # Plot reference line
    ax.axhline(0, color='gray', linestyle=':')
    
    # Plot Ichimoku if enabled
    if show_ichimoku:
        # Plot Tenkan and Kijun lines
        ax.plot(df["timestamp"], df["tenkan"], label="Tenkan-Sen", color='blue', linestyle='-')
        ax.plot(df["timestamp"], df["kijun"], label="Kijun-Sen", color='orange', linestyle='-')
        
        # Plot Cloud fill
        mask = ~df["senkou_a"].isna() & ~df["senkou_b"].isna()
        if mask.any():
            ax.fill_between(df["timestamp"][mask], 
                          df["senkou_a"][mask], df["senkou_b"][mask],
                          where=(df["senkou_a"][mask] >= df["senkou_b"][mask]),
                          interpolate=True, color='lightgreen', alpha=0.2, label="Kumo (Bullish)")
            
            ax.fill_between(df["timestamp"][mask], 
                          df["senkou_a"][mask], df["senkou_b"][mask],
                          where=(df["senkou_a"][mask] < df["senkou_b"][mask]),
                          interpolate=True, color='red', alpha=0.2, label="Kumo (Bearish)")
    
    # Plot SuperTrend if enabled
    if show_supertrend:
        latest_trend = df["trend"].iloc[-1] if not df.empty else 1
        trend_color = 'lime' if latest_trend == 1 else 'red'
        ax.plot(df["timestamp"], df["supertrend"], color=trend_color, linewidth=2, label="SuperTrend")
        
        # Plot buy/sell markers
        buy_signals = df[df["buy_signal"]]
        sell_signals = df[df["sell_signal"]]
        
        ax.scatter(buy_signals["timestamp"], buy_signals["msi"], 
                 marker="^", s=100, color="green", label="Buy Signal")
        ax.scatter(sell_signals["timestamp"], sell_signals["msi"], 
                 marker="v", s=100, color="red", label="Sell Signal")
    
    ax.set_title("📊 MSI Volatility Tracker")
    ax.legend()
    
    st.pyplot(fig)

def signal_statistics(df):
    """Display signal statistics"""
    if len(df) < 10:
        st.warning("Need more data for signal statistics.")
        return
    
    # Count signals
    entry_count = df['entry_signal'].sum()
    exit_count = df['exit_signal'].sum()
    bounce_count = df['bounce_signal'].sum()
    
    # Calculate signal density (signals per round)
    total_rounds = len(df)
    signal_density = (entry_count + exit_count + bounce_count) / total_rounds
    
    # Calculate accuracy metrics if we have outcome data
    # (This is a placeholder - in a real system you'd track outcomes)
    accuracy = 0.0
    if 'signal_correct' in df.columns:
        accuracy = df['signal_correct'].mean()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entry Signals", f"{int(entry_count)}")
        st.metric("Signal Density", f"{signal_density:.2f} per round")
    with col2:
        st.metric("Exit Signals", f"{int(exit_count)}")
        if 'signal_correct' in df.columns:
            st.metric("Signal Accuracy", f"{accuracy:.1%}")
    with col3:
        st.metric("Bounce Signals", f"{int(bounce_count)}")

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
        st.session_state.rounds.append({
            "timestamp": datetime.now(),
            "multiplier": mult,
            "score": score_value
        })
        st.rerun()

# Display fast entry mode if enabled
if FAST_ENTRY_MODE:
    fast_entry_mode_ui(PINK_THRESHOLD)

# Convert rounds to DataFrame
df = pd.DataFrame(st.session_state.rounds)

# Main App Logic
if not df.empty:
    # Run analysis
    processed_df, processing_time = analyze_data(
        df, 
        pink_threshold=PINK_THRESHOLD, 
        window_size=WINDOW_SIZE,
        signal_period=SIGNAL_PERIOD,
        thre_sensitivity=THRE_SENSITIVITY
    )
    
    # Performance metrics
    with col_hud:
        st.metric("Rounds Recorded", len(df))
        st.metric("Processing Time", f"{processing_time*1000:.1f} ms")
        
        # Calculate latest metrics
        if len(processed_df) > 0:
            latest_thre = processed_df["thre_value"].iloc[-1]
            latest_slope = processed_df["thre_slope"].iloc[-1]
            st.metric("THRE Value", f"{latest_thre:.2f}", delta=f"{latest_slope:.2f}")
    
    # Plot MSI Chart
    plot_msi_chart(processed_df)
    
    # Plot TDI+THRE Panel
    with st.expander("📈 TDI Panel with THRE-Powered Arrows", expanded=True):
        plot_tdi_thre(processed_df)
        if show_details:
            signal_statistics(processed_df)
    
    # Show THRE Panel if enabled
    if show_thre:
        with st.expander("🔬 True Harmonic Resonance Engine (THRE)", expanded=False):
            thre_panel(processed_df, THRE_SENSITIVITY)
    
    # Recent rounds log
    with st.expander("📄 Review / Edit Recent Rounds", expanded=False):
        edited = st.data_editor(df.tail(30), use_container_width=True, num_rows="dynamic")
        if st.button("✅ Commit Edits"):
            st.session_state.rounds = edited.to_dict('records')
            st.rerun()

else:
    st.info("Enter at least 1 round to begin analysis.")
