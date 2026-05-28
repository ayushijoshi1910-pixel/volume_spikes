import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="Stock Volume Spike Detector",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Volume Spike Detection System")

st.write(
    "Analyze stock trading volume and detect unusual activity."
)

# ---------------------------------------------------
# STOCK LIST
# ---------------------------------------------------

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Wipro": "WIPRO.NS"
}

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Settings")

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

period = st.sidebar.selectbox(
    "Select Time Period",
    ["3mo", "6mo", "1y", "2y"]
)

z_threshold = st.sidebar.slider(
    "Outlier Threshold (Z-Score)",
    1.0,
    5.0,
    2.5
)

# ---------------------------------------------------
# FETCH DATA
# ---------------------------------------------------

ticker = stocks[selected_stock]

df = yf.download(
    ticker,
    period=period,
    auto_adjust=True
)

# ---------------------------------------------------
# FIX MULTI-INDEX COLUMNS
# ---------------------------------------------------

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

# ---------------------------------------------------
# CONVERT VOLUME TO SERIES
# ---------------------------------------------------

volume = df["Volume"].astype(float)

# ---------------------------------------------------
# VOLUME ANALYSIS
# ---------------------------------------------------

df["Volume_Mean"] = (
    volume
    .rolling(window=10)
    .mean()
)

df["Volume_STD"] = (
    volume
    .rolling(window=10)
    .std()
)

df["Z_Score"] = (
    (volume - df["Volume_Mean"])
    / df["Volume_STD"]
)

# ---------------------------------------------------
# DETECT SPIKES
# ---------------------------------------------------

df["Spike"] = np.where(
    df["Z_Score"] > z_threshold,
    "Unusual Activity",
    "Normal"
)

# ---------------------------------------------------
# SHOW STATUS
# ---------------------------------------------------

latest_status = df["Spike"].iloc[-1]

st.subheader("Latest Volume Status")

if latest_status == "Unusual Activity":
    st.error("🚨 Unusual Volume Spike Detected")
else:
    st.success("✅ Volume Activity Normal")

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Volume",
    f"{int(volume.iloc[-1]):,}"
)

col2.metric(
    "Average Volume",
    f"{int(volume.mean()):,}"
)

col3.metric(
    "Highest Volume",
    f"{int(volume.max()):,}"
)

# ---------------------------------------------------
# DATAFRAME
# ---------------------------------------------------

st.subheader("Volume Analysis Data")

st.dataframe(
    df[
        [
            "Date",
            "Close",
            "Volume",
            "Z_Score",
            "Spike"
        ]
    ].tail(20)
)

# ---------------------------------------------------
# VOLUME CHART
# ---------------------------------------------------

st.subheader("Volume Trend Visualization")

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    df["Date"],
    volume,
    label="Volume"
)

ax.plot(
    df["Date"],
    df["Volume_Mean"],
    label="Average Volume"
)

# Highlight spikes

spikes = df[
    df["Spike"] == "Unusual Activity"
]

ax.scatter(
    spikes["Date"],
    spikes["Volume"],
    s=100,
    label="Spike"
)

ax.set_title(f"{selected_stock} Volume Analysis")

ax.set_xlabel("Date")
ax.set_ylabel("Trading Volume")

ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# SPIKE TABLE
# ---------------------------------------------------

st.subheader("Detected Unusual Activities")

spike_table = df[
    df["Spike"] == "Unusual Activity"
]

if len(spike_table) > 0:

    st.dataframe(
        spike_table[
            [
                "Date",
                "Close",
                "Volume",
                "Z_Score"
            ]
        ]
    )

else:
    st.info("No unusual volume spikes detected.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.info(
    "Volume spikes are detected using Z-Score based outlier detection."
)
