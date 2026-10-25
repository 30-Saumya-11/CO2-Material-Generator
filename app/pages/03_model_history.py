import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Add src folder and workspace folder to PATH to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_config, load_model_registry, save_model_registry, load_model

st.set_page_config(
    page_title="CO2 Material Generator - Model History",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load config and model registry
config = load_config()
registry = load_model_registry()
active_version = registry.get("active_version")

st.markdown("<h1 class='page-title'>Model Version Tracking</h1>", unsafe_allow_html=True)
st.write(
    "Monitor versions, evaluate model progress over training cycles, "
    "view predicted vs. actual calibration curves, and roll back models to previous versions."
)

if not registry.get("versions"):
    st.info("No model versions found in the registry. Please run the training pipeline first.")
    st.stop()

# ----------------- TABLE OF VERSIONS -----------------
st.write("### 📜 Model Registry Log")

versions_list = []
for v_name, v_data in registry["versions"].items():
    fw = v_data.get("forward_metrics", {})
    inv = v_data.get("inverse_metrics", {})
    
    versions_list.append({
        "Version": v_name,
        "Status": "ACTIVE" if v_name == active_version else "INACTIVE",
        "Training Date": v_data.get("training_date", "N/A"),
        "Dataset Size": v_data.get("dataset_size", "N/A"),
        "Train R²": f"{fw.get('train_r2', 0.0):.4f}" if fw.get('train_r2') is not None else "N/A",
        "Test R²": f"{fw.get('test_r2', 0.0):.4f}" if fw.get('test_r2') is not None else "N/A",
        "Test MAPE": f"{fw.get('test_mape', 0.0):.2f}%" if fw.get('test_mape') is not None else "N/A",
        "Inverse Val R²": f"{inv.get('validation_r2', 0.0):.4f}" if inv.get('validation_r2') is not None else "N/A",
        "Inverse Type": inv.get("model_type", "N/A").upper()
    })

df_versions = pd.DataFrame(versions_list)
st.dataframe(df_versions, use_container_width=True, hide_index=True)

# ----------------- PROGRESSION CHARTS -----------------
st.write("")
st.write("### 📈 Model Progression Analysis")

progress_data = []
for v_name, v_data in registry["versions"].items():
    fw = v_data.get("forward_metrics", {})
    progress_data.append({
        "Version": v_name,
        "Dataset Size": v_data.get("dataset_size", 573),
        "Test R2": fw.get("test_r2", 0.0)
    })
df_prog = pd.DataFrame(progress_data)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(
        x=df_prog["Version"],
        y=df_prog["Test R2"],
        mode="lines+markers",
        line=dict(color="#10B981", width=3),
        marker=dict(size=8, color="#3B82F6"),
        hovertemplate="Version: %{x}<br>Test R²: %{y:.4f}<extra></extra>"
    ))
    fig_r2.update_layout(
        title="Forward Model Test R² over Versions",
        xaxis_title="Version",
        yaxis_title="R² Score",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF')
    )
    fig_r2.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig_r2.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig_r2, use_container_width=True)

with col_chart2:
    fig_size = go.Figure()
    fig_size.add_trace(go.Scatter(
        x=df_prog["Version"],
        y=df_prog["Dataset Size"],
        mode="lines+markers",
        line=dict(color="#3B82F6", width=3, dash='dash'),
        marker=dict(size=8, color="#10B981"),
        hovertemplate="Version: %{x}<br>Dataset Size: %{y}<extra></extra>"
    ))
    fig_size.update_layout(
        title="Dataset Size Growth over Versions",
        xaxis_title="Version",
        yaxis_title="Number of Materials",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF')
    )
    fig_size.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig_size.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig_size, use_container_width=True)

# ----------------- CALIBRATION SCATTER -----------------
st.write("")
st.write("### 🎯 Model Calibration: Predicted vs. Measured CO₂")

try:
    with st.spinner("Generating calibration scatter plot..."):
        # Load dataset
        raw_path = config['paths']['raw_dataset']
        if not os.path.exists(raw_path) and os.path.exists(os.path.join("..", raw_path)):
            raw_path = os.path.join("..", raw_path)
            
        raw_df = pd.read_csv(raw_path)
        
        # Standardize columns
        col_mapping = {c: c.strip().upper() for c in raw_df.columns}
        for k, v in col_mapping.items():
            if v == 'CO2' or 'CO2' in v or 'CO₂' in v:
                col_mapping[k] = 'CO2'
        raw_df = raw_df.rename(columns=col_mapping)
        
        # Check if combined dataset exists to show the complete dataset calibration
        comb_path = config['paths']['combined_dataset']
        if not os.path.exists(comb_path) and os.path.exists(os.path.join("..", comb_path)):
            comb_path = os.path.join("..", comb_path)
            
        if os.path.exists(comb_path):
            st.caption("Displaying calibration over full dataset (baseline + new experiments)")
            plot_df = pd.read_csv(comb_path)
            plot_df = plot_df.rename(columns=col_mapping)
        else:
            st.caption("Displaying calibration over baseline raw dataset")
            plot_df = raw_df.copy()
            
        plot_df = plot_df[config['columns']].dropna()
        
        # Load active models
        active_forward = load_model("forward_model", active_version)
        active_scaler = load_model("scaler", active_version)
        
        input_cols = config['columns'][:-1]
        target_col = config['columns'][-1]
        
        X_raw = plot_df[input_cols].values
        y_actual = plot_df[target_col].values
        X_scaled = active_scaler.transform(X_raw)
        y_pred = active_forward.predict(X_scaled)
        
        indices = np.arange(len(plot_df))
        
        # Highlight points
        # If we have combined data, the new points are indices >= len(raw_df)
        if len(plot_df) > len(raw_df):
            baseline_idx = indices[:len(raw_df)]
            new_idx = indices[len(raw_df):]
        else:
            baseline_idx = indices
            new_idx = []
            
        # Split baseline into train/test splits for better visualization
        # We simulate the split on the baseline indices
        train_idx, test_idx = train_test_split(baseline_idx, test_size=0.15, random_state=42)
        
        fig_scatter = go.Figure()
        
        # Baseline Train
        fig_scatter.add_trace(go.Scatter(
            x=y_actual[train_idx],
            y=y_pred[train_idx],
            mode='markers',
            marker=dict(color='rgba(59, 130, 246, 0.6)', size=6),
            name='Baseline Train (85%)',
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
        ))
        
        # Baseline Test
        fig_scatter.add_trace(go.Scatter(
            x=y_actual[test_idx],
            y=y_pred[test_idx],
            mode='markers',
            marker=dict(color='rgba(245, 158, 11, 0.9)', size=8, line=dict(width=1, color='white')),
            name='Baseline Test (15%)',
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
        ))
        
        # New points (Active learning acquisitions)
        if len(new_idx) > 0:
            fig_scatter.add_trace(go.Scatter(
                x=y_actual[new_idx],
                y=y_pred[new_idx],
                mode='markers',
                marker=dict(color='rgba(16, 185, 129, 1.0)', size=9, symbol='star', line=dict(width=1, color='white')),
                name='New Active Learning Data',
                hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
            ))
            
        # 1:1 diagonal line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dash'),
            name='Ideal Calibration (1:1)'
        ))
        
        fig_scatter.update_layout(
            title=f"Predicted vs. Measured CO₂ Adsorption Capacity ({active_version})",
            xaxis_title="Measured CO₂ Adsorption (mmol/g)",
            yaxis_title="Predicted CO₂ Adsorption (mmol/g)",
            height=480,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            legend=dict(x=0.05, y=0.95, bgcolor='rgba(0,0,0,0.5)')
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig_scatter.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig_scatter, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load scatter plot: {str(e)}")

# ----------------- ROLLBACK SYSTEM -----------------
st.write("")
st.write("---")
st.write("### ⏪ Model Version Rollback")
st.write(
    "If a newer version of the model is currently active but underperforming in practice, "
    "you can switch the active deployment back to any previous version listed in the registry."
)

version_options = list(registry["versions"].keys())

col_rollback_sel, col_rollback_btn = st.columns([2, 1])

with col_rollback_sel:
    selected_rollback = st.selectbox(
        "Select model version to activate:",
        options=version_options,
        index=version_options.index(active_version) if active_version in version_options else 0
    )

with col_rollback_btn:
    st.write("")
    rollback_btn = st.button("Rollback Active Model Version", use_container_width=True)
    
    if rollback_btn:
        if selected_rollback == active_version:
            st.info(f"Version {selected_rollback} is already active.")
        else:
            with st.spinner("Switching active model deployment..."):
                try:
                    registry["active_version"] = selected_rollback
                    save_model_registry(registry)
                    st.success(f"Active model version changed to {selected_rollback} successfully!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during rollback: {str(e)}")
