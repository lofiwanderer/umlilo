import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths, butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
import pywt
import math

# ================== ENHANCED MORLET PHASE DETECTION SYSTEM ==================

@st.cache_data
def enhanced_morlet_wavelet_power(scores, wavelet='cmor', max_scale=64, min_scale=1):
    """
    Enhanced Morlet wavelet transform with improved power spectrum computation
    
    Args:
        scores: Time series data
        wavelet: Wavelet type (default: 'cmor' - Complex Morlet)
        max_scale: Maximum scale for analysis
        min_scale: Minimum scale for analysis
    
    Returns:
        coeffs: Wavelet coefficients
        power: Power spectrum
        freqs: Corresponding frequencies
        scales: Scales used for transform
    """
    # Use more refined scales for better resolution
    scales = np.arange(min_scale, max_scale)
    
    # Compute CWT with complex Morlet wavelet
    coeffs, freqs = pywt.cwt(scores, scales, wavelet)
    
    # Compute power (squared magnitude of coefficients)
    power = np.abs(coeffs) ** 2
    
    # Extract phase information from complex coefficients
    phase = np.angle(coeffs)
    
    return coeffs, power, freqs, scales, phase


def smooth_array(array, window=5):
    """
    Smooth an array using Savitzky-Golay filter
    
    Args:
        array: Input array to smooth
        window: Window size (must be odd)
    
    Returns:
        Smoothed array
    """
    if len(array) < window:
        return array
    
    # Ensure window is odd
    window = window if window % 2 == 1 else window + 1
    
    # Apply Savitzky-Golay filter for smooth derivatives
    return savgol_filter(array, window, 3)


@st.cache_data
def detect_morlet_phases(mean_energy, threshold_factor=0.5):
    """
    Detect wave phases from Morlet wavelet energy
    
    Args:
        mean_energy: Mean energy across scales
        threshold_factor: Factor for peak detection threshold
    
    Returns:
        List of dictionaries with round indices and corresponding phases
    """
    if len(mean_energy) < 10:
        return []
    
    # Smooth the energy curve for more stable derivatives
    energy_smooth = smooth_array(mean_energy, window=5)
    
    # Compute 1st derivative (slope)
    slope = np.gradient(energy_smooth)
    slope_smooth = smooth_array(slope, window=5)
    
    # Compute 2nd derivative (acceleration/curvature)
    accel = np.gradient(slope_smooth)
    accel_smooth = smooth_array(accel, window=5)
    
    # Find peaks in energy
    height_threshold = np.mean(energy_smooth) + threshold_factor * np.std(energy_smooth)
    peaks, peak_properties = find_peaks(energy_smooth, height=height_threshold, distance=3)
    
    # Calculate peak widths for phase length estimation
    if len(peaks) > 0:
        widths, heights, left_ips, right_ips = peak_widths(energy_smooth, peaks, rel_height=0.5)
    else:
        widths, heights, left_ips, right_ips = [], [], [], []
    
    # Initialize phase detection
    phases = []
    phase_status = "End Phase"  # Default phase
    
    for i in range(len(energy_smooth)):
        # Determine phase based on slope, acceleration, and peak proximity
        if len(peaks) > 0:
            # Find the nearest peak (future or past)
            nearest_peak_idx = np.argmin(np.abs(np.array(peaks) - i))
            nearest_peak = peaks[nearest_peak_idx]
            distance_to_peak = i - nearest_peak
            
            # If we have width information for this peak
            if nearest_peak_idx < len(widths):
                peak_width = widths[nearest_peak_idx]
                left_ip = left_ips[nearest_peak_idx]
                right_ip = right_ips[nearest_peak_idx]
                
                # Determine phase based on position relative to peak and inflection points
                if i < left_ip:  # Before left inflection point
                    if slope_smooth[i] > 0 and energy_smooth[i] < height_threshold:
                        phase_status = "Birth Phase"
                    elif slope_smooth[i] > 0:
                        phase_status = "Ascent Phase"
                elif left_ip <= i <= nearest_peak:  # Between left inflection and peak
                    phase_status = "Peak Phase"
                elif nearest_peak < i <= right_ip:  # Between peak and right inflection
                    phase_status = "Post-Peak"
                elif i > right_ip:  # After right inflection
                    if slope_smooth[i] < 0 and accel_smooth[i] < 0:
                        phase_status = "Falling Phase"
                    else:
                        phase_status = "End Phase"
            else:
                # If we can't use width info, use simpler heuristics
                if distance_to_peak < 0:  # Before peak
                    if slope_smooth[i] > 0 and energy_smooth[i] < height_threshold:
                        phase_status = "Birth Phase"
                    elif slope_smooth[i] > 0:
                        phase_status = "Ascent Phase"
                    elif slope_smooth[i] > 0 and energy_smooth[i] >= height_threshold:
                        phase_status = "Peak Phase"
                elif distance_to_peak == 0:  # At peak
                    phase_status = "Peak Phase"
                else:  # After peak
                    if slope_smooth[i] < 0 and np.abs(distance_to_peak) <= 3:
                        phase_status = "Post-Peak"
                    elif slope_smooth[i] < 0:
                        phase_status = "Falling Phase"
                    else:
                        phase_status = "End Phase"
        else:
            # No peaks detected, use only slope and acceleration
            if slope_smooth[i] > 0 and accel_smooth[i] > 0:
                phase_status = "Birth Phase"
            elif slope_smooth[i] > 0 and accel_smooth[i] <= 0:
                phase_status = "Ascent Phase"
            elif slope_smooth[i] <= 0 and accel_smooth[i] < 0:
                phase_status = "Falling Phase"
            else:
                phase_status = "End Phase"
        
        phases.append({"round": i, "phase": phase_status})
    
    return phases


# ================== VISUALIZATION COMPONENTS ==================

def plot_morlet_phases(energy, phases, timestamps=None):
    """
    Create Plotly visualization with phase highlighting
    
    Args:
        energy: Mean energy array
        phases: List of phase dictionaries
        timestamps: Optional timestamp array (x-axis)
    
    Returns:
        Plotly figure object
    """
    if timestamps is None:
        timestamps = np.arange(len(energy))
    
    # Create figure
    fig = go.Figure()
    
    # Add energy line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=energy,
        mode='lines',
        line=dict(color='purple', width=2),
        name='Energy',
        hoverinfo='y+name'
    ))
    
    # Phase color mapping
    phase_colors = {
        "Birth Phase": "rgba(135, 206, 250, 0.3)",    # Light blue
        "Ascent Phase": "rgba(144, 238, 144, 0.3)",  # Light green
        "Peak Phase": "rgba(255, 215, 0, 0.3)",      # Gold
        "Post-Peak": "rgba(255, 165, 0, 0.3)",      # Orange
        "Falling Phase": "rgba(255, 99, 71, 0.3)",   # Tomato
        "End Phase": "rgba(169, 169, 169, 0.3)"      # Gray
    }
    
    # Phase emoji mapping
    phase_emojis = {
        "Birth Phase": "🌱",
        "Ascent Phase": "🔼", 
        "Peak Phase": "💥",
        "Post-Peak": "🔄",
        "Falling Phase": "⚠️",
        "End Phase": "⏹️"
    }
    
    # Extract phases and create shaded regions
    current_phase = None
    start_idx = 0
    
    for i, phase_info in enumerate(phases):
        phase = phase_info["phase"]
        
        # When phase changes or we reach the end
        if current_phase != phase or i == len(phases) - 1:
            if current_phase is not None:
                # Add shaded region for the phase
                end_idx = i-1 if i < len(phases) else len(phases) - 1
                
                # Add shape (shaded background)
                fig.add_shape(
                    type="rect",
                    x0=timestamps[start_idx],
                    x1=timestamps[end_idx],
                    y0=0,
                    y1=max(energy) * 1.1,
                    fillcolor=phase_colors.get(current_phase, "rgba(211, 211, 211, 0.3)"),
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add phase annotation
                fig.add_annotation(
                    x=timestamps[start_idx + (end_idx - start_idx) // 2],
                    y=max(energy) * 0.9,
                    text=f"{phase_emojis.get(current_phase, '❓')} {current_phase}",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)"
                )
            
            # Update for next phase
            current_phase = phase
            start_idx = i
    
    # Add first derivative (slope)
    if len(energy) > 1:
        slope = smooth_array(np.gradient(energy), window=5)
        scaled_slope = slope * (max(energy) / max(np.abs(slope)) * 0.5) if max(np.abs(slope)) > 0 else slope
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scaled_slope,
            mode='lines',
            line=dict(color='blue', width=1, dash='dot'),
            name='Slope (scaled)',
            hoverinfo='y+name'
        ))
        
        # Add zero line for slope
        fig.add_shape(
            type="line",
            x0=min(timestamps),
            x1=max(timestamps),
            y0=0,
            y1=0,
            line=dict(color="gray", width=0.5, dash="dash"),
            layer="below"
        )
    
    # Layout configuration
    fig.update_layout(
        title="Morlet Wavelet Phase Detection",
        xaxis_title="Round",
        yaxis_title="Energy",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    # Add more detailed hover information
    fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Round: %{x}<br>Value: %{y:.3f}")
    
    return fig


# ================== ADVANCED PHASE ANALYSIS FUNCTIONS ==================

def compute_phase_metrics(phases):
    """
    Compute metrics about the detected phases
    
    Args:
        phases: List of phase dictionaries
    
    Returns:
        Dictionary with phase metrics
    """
    if not phases:
        return {}
    
    phase_counts = {}
    phase_durations = {}
    current_phase = None
    phase_start = 0
    
    for i, phase_info in enumerate(phases):
        phase = phase_info["phase"]
        
        # Count phases
        if phase not in phase_counts:
            phase_counts[phase] = 0
        phase_counts[phase] += 1
        
        # Track phase transitions for duration calculation
        if phase != current_phase:
            if current_phase is not None:
                if current_phase not in phase_durations:
                    phase_durations[current_phase] = []
                phase_durations[current_phase].append(i - phase_start)
            current_phase = phase
            phase_start = i
    
    # Add the last phase duration
    if current_phase is not None:
        if current_phase not in phase_durations:
            phase_durations[current_phase] = []
        phase_durations[current_phase].append(len(phases) - phase_start)
    
    # Calculate average durations
    avg_durations = {}
    for phase, durations in phase_durations.items():
        if durations:
            avg_durations[phase] = sum(durations) / len(durations)
    
    return {
        "phase_counts": phase_counts,
        "avg_durations": avg_durations,
        "phase_sequences": get_phase_sequences(phases)
    }


def get_phase_sequences(phases, min_length=3):
    """
    Extract common phase sequences
    
    Args:
        phases: List of phase dictionaries
        min_length: Minimum sequence length to consider
    
    Returns:
        List of common phase sequences
    """
    if not phases or len(phases) < min_length:
        return []
    
    # Extract just the phase values
    phase_values = [p["phase"] for p in phases]
    
    # Extract all sequences of length min_length to min_length+2
    sequences = {}
    max_seq_len = min(min_length + 2, len(phase_values))
    
    for seq_len in range(min_length, max_seq_len + 1):
        for i in range(len(phase_values) - seq_len + 1):
            seq = tuple(phase_values[i:i+seq_len])
            if seq not in sequences:
                sequences[seq] = 0
            sequences[seq] += 1
    
    # Filter for sequences that appear multiple times
    common_sequences = []
    for seq, count in sequences.items():
        if count > 1:
            common_sequences.append({
                "sequence": seq,
                "count": count,
                "confidence": round(count / (len(phase_values) - len(seq) + 1), 2)
            })
    
    # Sort by confidence
    common_sequences.sort(key=lambda x: x["confidence"], reverse=True)
    
    return common_sequences[:5]  # Return top 5


def predict_next_phase(phases, lookback=5):
    """
    Predict the next phase based on recent phase history
    
    Args:
        phases: List of phase dictionaries
        lookback: Number of recent phases to consider
    
    Returns:
        Dict with predicted next phase and confidence
    """
    if not phases or len(phases) < lookback + 1:
        return {"next_phase": "Unknown", "confidence": 0}
    
    # Extract phase values
    phase_values = [p["phase"] for p in phases]
    
    # Create a simple Markov model of phase transitions
    transitions = {}
    for i in range(len(phase_values) - 1):
        current = phase_values[i]
        next_phase = phase_values[i+1]
        
        if current not in transitions:
            transitions[current] = {}
        
        if next_phase not in transitions[current]:
            transitions[current][next_phase] = 0
        
        transitions[current][next_phase] += 1
    
    # Get current phase (last in the list)
    current_phase = phase_values[-1]
    
    if current_phase in transitions:
        # Find most likely next phase
        next_phases = transitions[current_phase]
        if next_phases:
            next_phase = max(next_phases.items(), key=lambda x: x[1])
            total = sum(next_phases.values())
            confidence = next_phase[1] / total if total > 0 else 0
            return {"next_phase": next_phase[0], "confidence": round(confidence, 2)}
    
    return {"next_phase": "Unknown", "confidence": 0}


# ================== STREAMLIT COMPONENTS ==================

def morlet_phase_panel(df, scores_col="score"):
    """
    Complete Streamlit panel for Morlet wavelet phase analysis
    
    Args:
        df: DataFrame with time series data
        scores_col: Column name for scores
    """
    st.subheader("🌊 Enhanced Morlet Phase Detection")
    
    # Extract scores
    scores = df[scores_col].fillna(0).values
    timestamps = df["timestamp"] if "timestamp" in df.columns else np.arange(len(scores))
    
    if len(scores) < 10:
        st.warning("Need at least 10 rounds to perform Morlet phase analysis.")
        return
    
    # Compute wavelet transform
    with st.spinner("Computing wavelet transform..."):
        coeffs, power, freqs, scales, phase = enhanced_morlet_wavelet_power(scores)
    
    # Compute mean energy across scales
    mean_energy = np.mean(power, axis=0)
    
    # Detect phases
    with st.spinner("Detecting wave phases..."):
        phases = detect_morlet_phases(mean_energy)
    
    # Create phase visualization
    fig = plot_morlet_phases(mean_energy, phases, timestamps)
    st.plotly_chart(fig, use_container_width=True)
    
    # Phase analytics
    with st.expander("🔍 Phase Analytics", expanded=False):
        metrics = compute_phase_metrics(phases)
        
        # Phase distribution
        if "phase_counts" in metrics and metrics["phase_counts"]:
            st.subheader("Phase Distribution")
            phase_df = pd.DataFrame({
                "Phase": list(metrics["phase_counts"].keys()),
                "Count": list(metrics["phase_counts"].values())
            })
            st.bar_chart(phase_df.set_index("Phase"))
        
        # Average durations
        if "avg_durations" in metrics and metrics["avg_durations"]:
            st.subheader("Average Phase Durations (rounds)")
            for phase, duration in metrics["avg_durations"].items():
                st.metric(phase, f"{duration:.1f} rounds")
        
        # Common sequences
        if "phase_sequences" in metrics and metrics["phase_sequences"]:
            st.subheader("Common Phase Sequences")
            for seq in metrics["phase_sequences"]:
                st.write(f"**{' → '.join(seq['sequence'])}** (Confidence: {seq['confidence']*100:.0f}%)")
    
    # Current phase status and prediction
    st.subheader("📡 Live Phase Status")
    col1, col2 = st.columns(2)
    
    current_phase = phases[-1]["phase"] if phases else "Unknown"
    with col1:
        phase_emoji = {
            "Birth Phase": "🌱", 
            "Ascent Phase": "🔼", 
            "Peak Phase": "💥", 
            "Post-Peak": "🔄", 
            "Falling Phase": "⚠️", 
            "End Phase": "⏹️"
        }
        st.metric(
            "Current Phase", 
            f"{phase_emoji.get(current_phase, '❓')} {current_phase}"
        )
    
    with col2:
        next_prediction = predict_next_phase(phases)
        confidence_color = "green" if next_prediction["confidence"] > 0.7 else \
                         "orange" if next_prediction["confidence"] > 0.4 else "red"
        
        next_phase = next_prediction["next_phase"]
        next_emoji = phase_emoji.get(next_phase, "❓")
        
        st.metric(
            "Predicted Next Phase",
            f"{next_emoji} {next_phase}",
            delta=f"Confidence: {next_prediction['confidence']*100:.0f}%"
        )

    # Decision guidance based on current phase
    action_map = {
        "Birth Phase": {"action": "MONITOR", "description": "Energy forming - watch for curve trajectory"},
        "Ascent Phase": {"action": "SCOUT", "description": "Energy rising - prepare entry strategy"},
        "Peak Phase": {"action": "ENTRY", "description": "Maximum energy - optimal entry window"},
        "Post-Peak": {"action": "CAUTION", "description": "Energy plateauing - final entry opportunity"},
        "Falling Phase": {"action": "AVOID", "description": "Energy declining - high risk zone"},
        "End Phase": {"action": "WAIT", "description": "Low energy - await new cycle formation"}
    }
    
    action = action_map.get(current_phase, {"action": "UNKNOWN", "description": "Cannot determine action"})
    
    st.info(
        f"**DECISION GUIDANCE:** {action['action']} — {action['description']}"
    )
    
    # Wave energy chart
    st.subheader("🔍 Wavelet Power Spectrum")
    power_fig = go.Figure()
    
    # Create heatmap of power spectrum
    heatmap = go.Heatmap(
        z=power,
        x=np.arange(len(scores)),
        y=scales,
        colorscale="Viridis",
        hovertemplate='Scale: %{y}<br>Round: %{x}<br>Power: %{z:.2f}'
    )
    
    power_fig.add_trace(heatmap)
    
    # Mark the phase transitions
    phase_changes = []
    last_phase = None
    for i, p in enumerate(phases):
        if p["phase"] != last_phase:
            phase_changes.append(i)
            last_phase = p["phase"]
    
    for change_idx in phase_changes:
        power_fig.add_shape(
            type="line",
            x0=change_idx,
            x1=change_idx,
            y0=0,
            y1=max(scales),
            line=dict(color="white", width=1, dash="dash")
        )
    
    power_fig.update_layout(
        title="Wavelet Power Spectrum with Phase Transitions",
        xaxis_title="Round",
        yaxis_title="Scale (lower = higher frequency)",
        yaxis_autorange="reversed",  # Higher scales at bottom
        height=400
    )
    
    st.plotly_chart(power_fig, use_container_width=True)
