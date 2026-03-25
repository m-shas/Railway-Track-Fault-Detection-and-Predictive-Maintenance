"""
Railway Track Fault Detection - Streamlit Dashboard
Secondary deliverable: Interactive dashboard with Plotly charts.

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Streamlit or Plotly not installed. Run: pip install streamlit plotly")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH = os.path.join(OUTPUTS_DIR, "dashboard_data.json")

st.set_page_config(
    page_title="Railway Track Fault Detection",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data
def load_dashboard_data():
    """Load dashboard_data.json produced by pipeline.py."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dashboard data not found: {DATA_PATH}")
        st.info("Run `python src/pipeline.py` first to generate the data.")
        st.stop()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_alert_log():
    """Load the alert log CSV."""
    path = os.path.join(OUTPUTS_DIR, "alert_log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


data = load_dashboard_data()
alert_df = load_alert_log()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("Railway AI Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Vibration Analysis", "Track Blocks", "AI Models", "Alerts"],
    index=0,
)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

block_ids = sorted(set(b["block_id"] for b in data["block_summary"]))
selected_blocks = st.sidebar.multiselect("Track Blocks", block_ids, default=block_ids)

location_ids = sorted(set(l["location_id"] for l in data["location_summary"]))
selected_locations = st.sidebar.multiselect("Locations", location_ids, default=location_ids)

alert_level_filter = st.sidebar.selectbox("Alert Level", ["ALL", "CRITICAL", "WARNING"])


# ── HELPERS ───────────────────────────────────────────────────────────────────

def health_color(score):
    if score >= 70:
        return "#10b981"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Overview")

    bs = [b for b in data["block_summary"] if b["block_id"] in selected_blocks]

    if not bs:
        st.warning("No blocks selected.")
        st.stop()

    avg_health = np.mean([b["avg_health"] for b in bs])
    total_crit = sum(b["n_critical"] for b in bs)
    total_warn = sum(b["n_warning"] for b in bs)
    total_anom = sum(b["n_anomalies"] for b in bs)
    avg_rul = np.mean([b["avg_rul"] for b in bs])

    # KPI cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Track Blocks", len(bs))
    c2.metric("Avg Health", f"{avg_health:.1f}")
    c3.metric("Critical", total_crit)
    c4.metric("Warnings", total_warn)
    c5.metric("Anomalies", total_anom)
    c6.metric("Avg RUL", f"{avg_rul:.0f} d")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Alert donut
        ad = data["alert_distribution"]
        fig = px.pie(names=list(ad.keys()), values=list(ad.values()),
                     title="Alert Distribution", hole=0.4,
                     color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fd = data["failure_distribution"]
        fig = px.bar(x=list(fd.keys()), y=list(fd.values()),
                     title="Failure Type Distribution",
                     color=list(fd.keys()), color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        ls = [l for l in data["location_summary"] if l["location_id"] in selected_locations]
        df_loc = pd.DataFrame(ls)
        if not df_loc.empty:
            fig = px.bar(df_loc, x="location_id", y="avg_health",
                         title="Location Health", color="avg_health",
                         color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        md = data["maintenance_distribution"]
        fig = px.pie(names=list(md.keys()), values=list(md.values()),
                     title="Maintenance Actions", hole=0.4)
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Health grid
    st.subheader("Track Block Health Grid")
    cols = st.columns(min(10, len(bs)))
    for i, b in enumerate(bs):
        with cols[i % len(cols)]:
            color = health_color(b["avg_health"])
            st.markdown(
                f'<div style="background:{color}22;border:2px solid {color};'
                f'border-radius:8px;text-align:center;padding:12px;margin:4px">'
                f'<b>{b["block_id"]}</b><br><span style="font-size:1.4em;color:{color}">'
                f'{b["avg_health"]}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: VIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Vibration Analysis":
    st.title("Vibration Analysis")

    va = data["vibration_accel"]

    col1, col2 = st.columns(2)

    with col1:
        df_accel = pd.DataFrame({
            "index": range(len(va["accel_magnitude"])),
            "magnitude": va["accel_magnitude"],
            "anomaly": va["vibr_anomaly"],
        })
        fig = px.scatter(df_accel, x="index", y="magnitude", color="anomaly",
                         title="Acceleration Magnitude (Anomalies in Red)",
                         color_continuous_scale=["#3b82f6", "#ef4444"])
        fig.update_traces(marker_size=2)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "temp" in va and "hum" in va:
            fig = px.scatter(x=va["temp"], y=va["hum"],
                             title="Temperature vs Humidity",
                             labels={"x": "Temperature", "y": "Humidity"},
                             opacity=0.4)
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Temp/Humidity data not available in vibration XLSX.")

    col3, col4 = st.columns(2)

    with col3:
        ts = data["time_series"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(y=ts["vibration"][:300], name="Vibration",
                                 line=dict(color="#3b82f6", width=1)), secondary_y=False)
        fig.add_trace(go.Scatter(y=ts["failure_prob"][:300], name="Failure Prob",
                                 line=dict(color="#ef4444", width=1)), secondary_y=True)
        fig.update_layout(template="plotly_dark", height=400,
                          title="PLC Vibration & Failure Probability")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(va["accel_magnitude"], nbins=40,
                           title="Vibration Magnitude Histogram",
                           color_discrete_sequence=["#10b981"])
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: TRACK BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Track Blocks":
    st.title("Track Blocks")

    bs = [b for b in data["block_summary"] if b["block_id"] in selected_blocks]
    df_bs = pd.DataFrame(bs).sort_values("block_id")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df_bs, x="avg_rul", y="block_id", orientation="h",
                     title="RUL by Block", color="avg_rul",
                     color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df_bs, x="avg_vibration", y="block_id", orientation="h",
                     title="Vibration by Block",
                     color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Block Detail Table")
    st.dataframe(
        df_bs[["block_id", "avg_health", "avg_rul", "avg_vibration",
               "n_critical", "n_warning", "n_anomalies"]],
        use_container_width=True,
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AI MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "AI Models":
    st.title("AI Model Performance")

    mm = data["model_metrics"]

    # Metrics cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RUL MAE", f"{mm['rul']['mae']:.2f} days")
    c2.metric("RUL RMSE", f"{mm['rul']['rmse']:.2f} days")
    c3.metric("RUL R2", f"{mm['rul']['r2']:.4f}")
    c4.metric("Classifier Acc", f"{mm['classifier_accuracy']*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        rs = data["rul_scatter"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs["actual"], y=rs["predicted"], mode="markers",
                                 marker=dict(size=3, color="#3b82f6", opacity=0.5),
                                 name="Predictions"))
        max_val = max(max(rs["actual"]), max(rs["predicted"]))
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                 mode="lines", line=dict(dash="dash", color="#ef4444"),
                                 name="Perfect"))
        fig.update_layout(template="plotly_dark", height=450,
                          title="RUL: Actual vs Predicted",
                          xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fi = data["feature_importance"]
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        fig = px.bar(x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
                     orientation="h", title="Feature Importance",
                     color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    cm = mm.get("confusion_matrix", [])
    if cm:
        st.subheader("Confusion Matrix")
        labels = [f"C{i+1}" for i in range(len(cm))]
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues",
                        title="10-Class Confusion Matrix")
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Pipeline architecture
    st.subheader("Pipeline Architecture")
    st.code(
        "CSV + XLSX -> Preprocess -> Isolation Forest -> "
        "GradientBoosting (RUL) + RandomForest (Fault) -> Alert Engine -> Dashboard",
        language="text"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ALERTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Alerts":
    st.title("Alerts & Maintenance")

    alerts = data["alerts"]

    # Filter
    if alert_level_filter != "ALL":
        alerts = [a for a in alerts if a["level"] == alert_level_filter]

    st.metric("Total Alerts Shown", len(alerts))

    # Alert table
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(df_alerts, use_container_width=True, height=400)

        # Maintenance recommendation cards
        st.subheader("Top Maintenance Recommendations")
        for a in alerts[:6]:
            level_color = "#ef4444" if a["level"] == "CRITICAL" else "#f59e0b"
            st.markdown(
                f'<div style="background:#213243;border-left:4px solid {level_color};'
                f'border-radius:8px;padding:16px;margin-bottom:8px">'
                f'<b>{a["block_id"]} — {a["fault"]}</b><br>'
                f'<span style="color:#8899aa">{a["action"]}</span><br>'
                f'<span style="color:#3b82f6">Score: {a["score"]} | RUL: {a["rul"]}d</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Alert distribution bar
        st.subheader("Alert Level Distribution")
        level_counts = {}
        for a in data["alerts"]:
            level_counts[a["level"]] = level_counts.get(a["level"], 0) + 1
        fig = px.bar(x=list(level_counts.keys()), y=list(level_counts.values()),
                     color=list(level_counts.keys()),
                     color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b"},
                     title="Alert Counts by Level")
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts match the current filter.")
