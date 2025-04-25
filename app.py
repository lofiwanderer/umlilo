import streamlit as st
import streamlit as st
import matplotlib.pyplot as plt

# Initialize session state
if 'momentum_line' not in st.session_state:
    st.session_state.momentum_line = [0]
    st.session_state.rounds = []
    st.session_state.volatility_changes = 0

# Updated scoring function
def score_round(multiplier):
    if multiplier < 1.50:
        return -1.5  # Blue: 1.00x - 1.49x
    elif 1.50 <= multiplier < 2.00:
        return -1    # Blue: 1.50x - 1.99x
    elif 2.00 <= multiplier < 5.00:
        return 1     # Purple: 2.00x - 4.99x
    elif 5.00 <= multiplier < 10.00:
        return 1.5   # Purple: 5.00x - 9.99x
    elif 10.00 <= multiplier < 20.00:
        return 2     # Pink: 10.00x - 19.99x
    elif 20.00 <= multiplier < 50.00:
        return 3     # Pink: 20.00x - 49.99x
    else:
        return 4     # Pink: 50.00x and above

# UI layout
st.title("Aviator Momentum Tracker")
st.markdown("Enter your round multiplier to track momentum")

# Input section
multiplier = st.number_input("Enter Round Multiplier", min_value=1.00, max_value=999.99, step=0.01)
if st.button("Log Round"):
    st.session_state.rounds.append(multiplier)

# Process scoring and momentum
for i in range(len(st.session_state.rounds) - len(st.session_state.momentum_line) + 1):
    idx = len(st.session_state.momentum_line) - 1
    current = st.session_state.rounds[idx]
    delta = score_round(current)
    new_score = st.session_state.momentum_line[-1] + delta

    # Volatility check
    if len(st.session_state.momentum_line) >= 3:
        prev_diff = st.session_state.momentum_line[-1] - st.session_state.momentum_line[-2]
        new_diff = new_score - st.session_state.momentum_line[-1]
        if (prev_diff > 0 and new_diff < 0) or (prev_diff < 0 and new_diff > 0):
            st.session_state.volatility_changes += 1

    st.session_state.momentum_line.append(new_score)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(st.session_state.momentum_line, marker='o', linestyle='-', color='blue')
ax.axhline(y=0, color='black', linestyle='--')
ax.axhspan(6, 10, facecolor='green', alpha=0.2, label='Burst Zone')
ax.axhspan(0, 5, facecolor='lightgreen', alpha=0.2, label='Neutral-Positive')
ax.axhspan(-5, 0, facecolor='orange', alpha=0.2, label='Neutral-Negative')
ax.axhspan(-10, -5, facecolor='red', alpha=0.2, label='Red Zone')
ax.set_title("Aviator Momentum Tracker")
ax.set_xlabel("Round")
ax.set_ylabel("Momentum Score")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Volatility tracker
st.subheader(f"Volatility Flips: {st.session_state.volatility_changes}")
if st.session_state.volatility_changes >= 6:
    st.error("HIGH Volatility — Avoid entries!")
elif st.session_state.volatility_changes >= 3:
    st.warning("MODERATE Volatility — Enter with caution.")
else:
    st.success("LOW Volatility — Stable play window.")

# Reset tracker
if st.button("Reset Tracker"):
    st.session_state.momentum_line = [0]
    st.session_state.rounds = []
    st.session_state.volatility_changes = 0

