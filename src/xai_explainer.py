"""
Railway Track Fault Detection - SHAP Explainable AI Module
Generates interactive Plotly charts from SHAP (SHapley Additive exPlanations)
for the Random Forest fault classifier.

Public API:
    compute_shap_values(rf_model, X_test_scaled, feature_names) -> dict
    build_shap_summary_fig(shap_dict)        -> go.Figure  (beeswarm-like)
    build_shap_bar_fig(shap_dict)            -> go.Figure  (mean |SHAP|)
    build_shap_waterfall_fig(shap_dict, idx) -> go.Figure  (single prediction)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# ── SHAP VALUE COMPUTATION ────────────────────────────────────────────────────

def compute_shap_values(
    rf_model,
    X_test_scaled: np.ndarray,
    feature_names: List[str],
    max_samples: int = 200,
) -> dict:
    """Compute SHAP values using TreeExplainer on a Random Forest.

    Parameters
    ----------
    rf_model : RandomForestClassifier
        Trained sklearn Random Forest.
    X_test_scaled : np.ndarray, shape (n_samples, n_features)
        Scaled test features.
    feature_names : list of str
        Feature column names.
    max_samples : int
        Max rows to compute SHAP for (keeps it fast).

    Returns
    -------
    dict with keys:
        shap_values     : np.ndarray (n_samples, n_features) — mean across classes
        feature_names   : list of str
        X_sample        : np.ndarray — the input rows used
        base_value      : float — expected model output
        class_names     : list of str
    """
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError("SHAP not installed. Run: pip install shap")

    X_sample = X_test_scaled[:max_samples]

    explainer = shap.TreeExplainer(rf_model)
    raw = explainer.shap_values(X_sample)

    # ── Handle different SHAP API return shapes ────────────────────────────
    # Old SHAP  (<0.40): list of (n_samples, n_features) arrays — one per class
    # New SHAP (>=0.42): 3D numpy array of shape (n_samples, n_features, n_classes)

    if isinstance(raw, list):
        # List format: raw[class_idx] -> (n_samples, n_features)
        raw_arr = np.array(raw)                           # (n_classes, n_samples, n_features)
        # Mean |SHAP| across classes → (n_samples, n_features)
        shap_matrix = np.mean(np.abs(raw_arr), axis=0)
        # Directed: use each sample's predicted class SHAP values
        predicted_classes = rf_model.predict(X_sample)
        shap_directed = np.zeros_like(shap_matrix)
        for i, cls in enumerate(predicted_classes):
            shap_directed[i] = raw_arr[int(cls), i, :]

    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # New API: shape (n_samples, n_features, n_classes)
        # Mean |SHAP| across classes → (n_samples, n_features)
        shap_matrix = np.abs(raw).mean(axis=2)
        predicted_classes = rf_model.predict(X_sample)
        shap_directed = np.zeros((raw.shape[0], raw.shape[1]))
        for i, cls in enumerate(predicted_classes):
            shap_directed[i] = raw[i, :, int(cls)]

    else:
        # Binary or regression: raw is already (n_samples, n_features)
        shap_matrix   = np.abs(raw)
        shap_directed = raw

    class_names = [f"C{i+1}" for i in range(rf_model.n_classes_)]
    base_value = float(np.mean([e for e in explainer.expected_value])) \
        if isinstance(explainer.expected_value, (list, np.ndarray)) \
        else float(explainer.expected_value)

    return {
        "shap_values":     shap_directed,       # (n_samples, n_features) directed
        "shap_abs":        shap_matrix,          # (n_samples, n_features) abs mean across classes
        "feature_names":   feature_names,
        "X_sample":        X_sample,
        "base_value":      base_value,
        "class_names":     class_names,
        "n_samples":       len(X_sample),
    }


# ── PLOTLY CHART BUILDERS ─────────────────────────────────────────────────────

def build_shap_bar_fig(shap_dict: dict):
    """Build a horizontal bar chart of mean |SHAP| per feature (Plotly).

    This is the cleanest chart for a research paper.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        raise ImportError("Plotly not installed. Run: pip install plotly")

    mean_abs = np.abs(shap_dict["shap_values"]).mean(axis=0)
    feature_names = shap_dict["feature_names"]

    # Sort descending
    sorted_idx = np.argsort(mean_abs)
    sorted_feats = [feature_names[i] for i in sorted_idx]
    sorted_vals  = mean_abs[sorted_idx]

    # Colour gradient: low → grey, high → accent blue/orange
    max_val = sorted_vals.max() if sorted_vals.max() > 0 else 1.0
    colors  = [
        f"rgba(59,130,246,{0.3 + 0.7 * v / max_val:.2f})"
        for v in sorted_vals
    ]

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_feats,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in sorted_vals],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean |SHAP| = %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="SHAP Feature Importance (Mean |SHAP| Value)",
                   font=dict(size=16)),
        xaxis_title="Mean |SHAP| value (impact on model output)",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=60, t=50, b=40),
        plot_bgcolor="#213243",
        paper_bgcolor="#213243",
    )
    return fig


def build_shap_summary_fig(shap_dict: dict, max_points: int = 150):
    """Build a beeswarm-like scatter plot of SHAP values (Plotly).

    Each point = one sample × one feature.
    X = SHAP value (positive = pushes toward C1 fault)
    Y = feature (jittered)
    Color = normalized feature value (low=blue, high=red)
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        raise ImportError("Plotly not installed.")

    shap_vals = shap_dict["shap_values"][:max_points]   # (n_samples, n_features)
    X_sample  = shap_dict["X_sample"][:max_points]
    features  = shap_dict["feature_names"]
    n_features = len(features)

    traces = []
    for fi in range(n_features):
        sv = shap_vals[:, fi]
        fv = X_sample[:, fi]

        # Normalize feature values 0-1 for color
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / (fv_max - fv_min + 1e-9)

        # Jitter y position
        jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
        y_pos  = fi + jitter

        # Color: blue (low) → red (high)
        color_val = [
            f"rgba({int(255*v)},{int(50*(1-v))},{int(255*(1-v))},0.7)"
            for v in fv_norm
        ]

        traces.append(go.Scatter(
            x=sv,
            y=y_pos,
            mode="markers",
            marker=dict(size=4, color=color_val, line=dict(width=0)),
            name=features[fi],
            text=[f"{features[fi]}<br>SHAP={s:.4f} | Val={fv[j]:.3f}"
                  for j, s in enumerate(sv)],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="SHAP Summary Plot — Feature Impact Distribution",
                   font=dict(size=16)),
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_features)),
            ticktext=features,
            showgrid=True,
            gridcolor="#2d3f52",
        ),
        template="plotly_dark",
        height=max(500, n_features * 30),
        margin=dict(l=0, r=20, t=50, b=40),
        plot_bgcolor="#213243",
        paper_bgcolor="#213243",
        shapes=[dict(type="line", x0=0, x1=0,
                     y0=-0.5, y1=n_features - 0.5,
                     line=dict(color="#8899aa", width=1, dash="dash"))],
    )
    return fig


def build_shap_waterfall_fig(shap_dict: dict, sample_idx: int = 0,
                              predicted_label: str = None):
    """Build a waterfall chart explaining a single AI prediction (Plotly).

    Shows how each feature pushed the prediction above/below the baseline.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        raise ImportError("Plotly not installed.")

    sv = shap_dict["shap_values"][sample_idx]  # (n_features,)
    features = shap_dict["feature_names"]
    base = shap_dict["base_value"]
    xv   = shap_dict["X_sample"][sample_idx]

    # Sort by absolute SHAP → show top 12 features
    top_n = 12
    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    sv_top   = sv[sorted_idx]
    feat_top = [f"{features[i]}<br>(val={xv[i]:.3f})" for i in sorted_idx]

    # Cumulative for waterfall
    cumulative = base + np.cumsum(np.concatenate([[0], sv_top]))[:-1]

    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in sv_top]

    title_suffix = f" → Predicted: {predicted_label}" if predicted_label else ""

    fig = go.Figure()

    # Baseline
    fig.add_trace(go.Bar(
        x=["Baseline E[f(x)]"],
        y=[base],
        marker_color="#8b5cf6",
        name="Baseline",
        text=[f"E[f(x)] = {base:.3f}"],
        textposition="outside",
        hovertemplate="Baseline: %{y:.4f}<extra></extra>",
    ))

    # Feature contributions
    for i, (feat, val, cum, color) in enumerate(
            zip(feat_top, sv_top, cumulative, colors)):
        fig.add_trace(go.Bar(
            x=[feat],
            y=[abs(val)],
            base=[cum if val >= 0 else cum + val],
            marker_color=color,
            showlegend=False,
            text=[f"+{val:.4f}" if val >= 0 else f"{val:.4f}"],
            textposition="outside",
            hovertemplate=f"<b>{feat}</b><br>SHAP: {val:+.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"SHAP Waterfall — Single Prediction Explanation{title_suffix}",
                   font=dict(size=15)),
        xaxis_title="Feature",
        yaxis_title="Model Output (contribution)",
        barmode="stack",
        template="plotly_dark",
        height=480,
        margin=dict(l=0, r=20, t=60, b=80),
        plot_bgcolor="#213243",
        paper_bgcolor="#213243",
        showlegend=False,
    )
    return fig
