import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go

# Add src folder and workspace folder to PATH to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_config, load_model_registry, load_model
from generate_material import generate_candidates

st.set_page_config(
    page_title="CO2 Material Generator - Generate",
    page_icon="🌱",
    layout="wide"
)

# Custom Premium Styling
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
    
    .badge-safe {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #047857;
    }
    
    .badge-hypothesis {
        background-color: #78350F;
        color: #FDE68A;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #D97706;
    }
    
    .badge-speculative {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #DC2626;
    }
    
    .candidate-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .property-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .property-label {
        color: #9CA3AF;
        font-size: 0.9rem;
    }
    
    .property-value {
        font-weight: 600;
        color: #FFFFFF;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration and model registry
config = load_config()
registry = load_model_registry()
active_version = registry.get("active_version")

# Sidebar status
st.sidebar.markdown("### 🌱 Active ML Model Status")
if active_version and active_version in registry.get("versions", {}):
    version_data = registry["versions"][active_version]
    fw_metrics = version_data["forward_metrics"]
    dataset_size = version_data["dataset_size"]
    st.sidebar.markdown(f"**Active Version:** `{active_version}`")
    st.sidebar.markdown(f"**Dataset Size:** `{dataset_size} materials`")
    st.sidebar.markdown(f"**Forward Test $R^2$:** `{fw_metrics['test_r2']:.4f}`")
else:
    st.sidebar.markdown("**Active Version:** `None` (Train model first)")
    dataset_size = 573

st.markdown("<h1 class='page-title'>Porous Carbon Generator</h1>", unsafe_allow_html=True)

if not active_version:
    st.error("No active model version found. Please run the training pipeline first.")
    st.stop()

# ----------------- UI INPUTS -----------------
st.markdown("### 🛠️ Step 1: Input Targets and Synthesis Constraints")

# Slide configuration based on model bounds
co2_ranges = config['feature_ranges']['CO2']

col_slider, col_indicator = st.columns([3, 1])

with col_slider:
    target_co2 = st.slider(
        "Target CO₂ Adsorption Uptake (mmol/g)",
        min_value=0.1,
        max_value=15.0,
        value=5.0,
        step=0.1
    )

with col_indicator:
    # Render color indicator box immediately
    if 0.36 <= target_co2 <= 8.20:
        st.markdown(
            f"<div style='background-color:#065F46; padding: 1.25rem; border-radius: 8px; border: 1px solid #047857; text-align: center; height:100%'>"
            f"<span style='color:#A7F3D0; font-weight:bold; font-size:1.1rem;'>SAFE ZONE</span><br/>"
            f"<span style='color:#E6FFFA; font-size:0.85rem;'>Target is within training limits ({co2_ranges['min']} - {co2_ranges['max']})</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    elif 8.20 < target_co2 <= 9.84 or 0.1 <= target_co2 < 0.36:
        st.markdown(
            f"<div style='background-color:#78350F; padding: 1.25rem; border-radius: 8px; border: 1px solid #D97706; text-align: center; height:100%'>"
            f"<span style='color:#FDE68A; font-weight:bold; font-size:1.1rem;'>HYPOTHESIS ZONE</span><br/>"
            f"<span style='color:#FFFDF5; font-size:0.85rem;'>Extrapolation under 20%. Suggests lab validation.</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#7F1D1D; padding: 1.25rem; border-radius: 8px; border: 1px solid #DC2626; text-align: center; height:100%'>"
            f"<span style='color:#FECACA; font-weight:bold; font-size:1.1rem;'>SPECULATIVE ZONE</span><br/>"
            f"<span style='color:#FFF5F5; font-size:0.85rem;'>Extrapolation exceeds 20%. Predictions are highly speculative.</span>"
            f"</div>",
            unsafe_allow_html=True
        )

st.write("")

# Checkbox and input for locked features
col_temp, col_press = st.columns(2)

with col_temp:
    fix_t = st.checkbox("Fix Temperature constraint?")
    fixed_temp = None
    if fix_t:
        fixed_temp = st.selectbox(
            "Select Temperature (°C) to lock:",
            options=[0.0, 15.0, 25.0, 30.0],
            index=2
        )

with col_press:
    fix_p = st.checkbox("Fix Pressure constraint?")
    fixed_press = None
    if fix_p:
        fixed_press = st.slider(
            "Select Pressure (bar) to lock:",
            min_value=0.10,
            max_value=1.01,
            value=1.00,
            step=0.05
        )

st.write("")
generate_btn = st.button("Generate Porous Carbon Properties", type="primary")

# ----------------- INFERENCE PIPELINE -----------------
if generate_btn:
    with st.spinner("Generating and validating materials..."):
        try:
            candidates = generate_candidates(
                target_co2,
                fixed_temp=fixed_temp,
                fixed_press=fixed_press,
                version=active_version
            )
        except Exception as e:
            st.error(f"Error during material generation: {str(e)}")
            st.stop()
            
    if not candidates:
        st.warning("No material candidates were generated.")
        st.stop()
        
    # Get overall level from candidate 1
    conf_level = candidates[0]['confidence_level']
    conf_msg = candidates[0]['confidence_details']['message']
    
    st.markdown("### 🏆 Step 2: Generation Results")
    
    # Render Confidence Badge
    if conf_level == "SAFE":
        st.markdown(f"**Confidence Level:** <span class='badge-safe'>SAFE</span>", unsafe_allow_html=True)
        st.success(conf_msg)
    elif conf_level == "HYPOTHESIS":
        st.markdown(f"**Confidence Level:** <span class='badge-hypothesis'>HYPOTHESIS</span>", unsafe_allow_html=True)
        st.warning(conf_msg)
    else:
        st.markdown(f"**Confidence Level:** <span class='badge-speculative'>SPECULATIVE</span>", unsafe_allow_html=True)
        st.error(conf_msg)
        
    st.write("")
    
    # 3 Columns for Candidate Cards
    col_c1, col_c2, col_c3 = st.columns(3)
    cols = [col_c1, col_c2, col_c3]
    
    for i, candidate in enumerate(candidates):
        with cols[i]:
            st.markdown(
                f"<div class='candidate-card'>"
                f"<h3 style='margin-top:0; color:#3B82F6;'>Candidate {i+1}</h3>"
                f"<div class='property-row'><span class='property-label'>Validation CO₂</span>"
                f"<span class='property-value' style='color:#10B981;'>{candidate['validation_co2']:.3f} mmol/g</span></div>"
                f"<div class='property-row'><span class='property-label'>Validation Error</span>"
                f"<span class='property-value'>{candidate['validation_error_pct']:.2f}%</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Confidence Progress Bar
            conf_val = candidate['confidence_score']
            st.write("**Overall Confidence Score**")
            st.progress(conf_val / 100.0)
            st.caption(f"{conf_val:.1f}% Confidence")
            
            # Properties table
            props = candidate['properties']
            st.write("")
            st.write("**Recommended Properties:**")
            
            # Format nicely
            props_df = pd.DataFrame([
                {"Property": "Surface Area (SA)", "Value": f"{props['SA']:.1f} m²/g"},
                {"Property": "Total Pore Volume (TPV)", "Value": f"{props['TPV']:.3f} cm³/g"},
                {"Property": "Micropore Volume (MPV)", "Value": f"{props['MPV']:.3f} cm³/g"},
                {"Property": "Carbon Content (C)", "Value": f"{props['C']:.2f} %"},
                {"Property": "Hydrogen Content (H)", "Value": f"{props['H']:.2f} %"},
                {"Property": "Nitrogen Content (N)", "Value": f"{props['N']:.2f} %"},
                {"Property": "Oxygen Content (O)", "Value": f"{props['O']:.2f} %"},
                {"Property": "Temperature (T)", "Value": f"{props['T']:.2f} °C"},
                {"Property": "Pressure (P)", "Value": f"{props['P']:.2f} bar"}
            ])
            st.dataframe(props_df, use_container_width=True, hide_index=True)
            
    # ----------------- SHAP PLOTS -----------------
    st.write("")
    st.markdown("### 🔍 Step 3: Feature Attribution (SHAP Analysis)")
    st.write(
        "SHAP values measure the impact of each porous carbon property on the forward model's CO₂ uptake prediction. "
        "Positive values (green) drive adsorption capacity up; negative values (red) pull it down."
    )
    
    with st.spinner("Calculating SHAP feature attributions..."):
        try:
            # Load models
            forward_model = load_model("forward_model", active_version)
            scaler = load_model("scaler", active_version)
            explainer = load_model("forward_explainer", active_version)
            
            input_cols = config['columns'][:-1]
            
            # Use Candidate 1 properties
            cand1_properties = candidates[0]['properties']
            cand1_df = pd.DataFrame([cand1_properties])
            
            # Scale
            X_scaled = scaler.transform(cand1_df[input_cols].values)
            
            # Compute SHAP
            # Explainer could return an Explanation object or list of shap values
            try:
                shap_exp = explainer(X_scaled)
                if hasattr(shap_exp, 'values'):
                    shap_vals = shap_exp.values[0]
                else:
                    shap_vals = explainer.shap_values(X_scaled)[0]
            except Exception:
                try:
                    shap_vals = explainer.shap_values(X_scaled)[0]
                except Exception:
                    # Fallback to GBDT feature importance if shap explainer fails
                    shap_vals = forward_model.feature_importances_
                    
            # Sort shap values for visualization
            shap_dict = {feat: float(val) for feat, val in zip(input_cols, shap_vals)}
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            
            feats_sorted = [x[0] for x in sorted_shap]
            vals_sorted = [x[1] for x in sorted_shap]
            colors_sorted = ['#10B981' if v >= 0 else '#EF4444' for v in vals_sorted]
            
            # Plot using Plotly
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=feats_sorted,
                x=vals_sorted,
                orientation='h',
                marker_color=colors_sorted,
                hovertemplate="Feature: %{y}<br>SHAP Attribution: %{x:.4f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Feature Contributions to CO₂ Uptake (Candidate 1, Base Target = {target_co2:.2f})",
                xaxis_title="SHAP Value (Contribution)",
                yaxis_title="Carbon Material Property / Environment Variable",
                yaxis=dict(autorange="reverse"),
                height=450,
                margin=dict(l=100, r=20, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF')
            )
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"SHAP explanation could not be fully rendered: {str(e)}. Displaying GBDT Feature Importances instead.")
            # Fallback to feature importance plot
            try:
                importances = forward_model.feature_importances_
                importance_dict = {feat: float(val) for feat, val in zip(input_cols, importances)}
                sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                feats_sort = [x[0] for x in sorted_importances]
                vals_sort = [x[1] for x in sorted_importances]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=feats_sort,
                    x=vals_sort,
                    orientation='h',
                    marker_color='#3B82F6',
                    hovertemplate="Feature: %{y}<br>Gini Importance: %{x:.4f}<extra></extra>"
                ))
                fig.update_layout(
                    title="Forward GBDT Model Feature Importance (Gini)",
                    xaxis_title="Relative Gini Importance",
                    yaxis_title="Material Properties",
                    yaxis=dict(autorange="reverse"),
                    height=450,
                    margin=dict(l=100, r=20, t=50, b=50),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF')
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e_inner:
                st.error(f"Failed to load GBDT model fallback feature importance: {str(e_inner)}")
