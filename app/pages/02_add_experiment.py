import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add src folder and workspace folder to PATH to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_config, load_model_registry
from retrain import retrain_models

st.set_page_config(
    page_title="CO2 Material Generator - Add Experiment",
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

# Load configuration and model registry
config = load_config()
registry = load_model_registry()
active_version = registry.get("active_version")

st.markdown("<h1 class='page-title'>Log Experimental Validation Data</h1>", unsafe_allow_html=True)
st.write(
    "Log new experimental laboratory results here. Once logged, you can run the retraining loop "
    "to refit the models on the updated dataset. The system will compare the retrained models against the active version "
    "and automatically deploy them if performance improves."
)

st.write("### 🧪 Step 1: Input Experimental Properties")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🔬 Porosity Parameters")
    sa = st.number_input("Surface Area (SA, m²/g)", min_value=1.0, max_value=5000.0, value=1500.0, step=50.0)
    tpv = st.number_input("Total Pore Volume (TPV, cm³/g)", min_value=0.01, max_value=5.0, value=0.80, step=0.05)
    mpv = st.number_input("Micropore Volume (MPV, cm³/g)", min_value=0.01, max_value=3.0, value=0.45, step=0.05)

with col2:
    st.markdown("##### 🧬 Elemental Composition (%)")
    c_perc = st.number_input("Carbon Content (C, %)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    h_perc = st.number_input("Hydrogen Content (H, %)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
    n_perc = st.number_input("Nitrogen Content (N, %)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    o_perc = st.number_input("Oxygen Content (O, %)", min_value=0.0, max_value=100.0, value=10.5, step=0.5)

with col3:
    st.markdown("##### 🌡️ Environmental & Capture Adsorption")
    temp = st.number_input("Temperature (T, °C)", min_value=-50.0, max_value=200.0, value=25.0, step=5.0)
    press = st.number_input("Pressure (P, bar)", min_value=0.01, max_value=50.0, value=1.00, step=0.1)
    co2_uptake = st.number_input("Measured CO₂ Adsorption (mmol/g)", min_value=0.0, max_value=30.0, value=3.20, step=0.1)

notes = st.text_input("Experiment Notes / Source Reference (e.g. 'Lab Batch #12B', 'Namdeo et al.')", value="Lab Batch 1A")

submit_btn = st.button("Submit Result", type="primary")

# File path resolution
new_exps_path = config['paths']['new_experiments']
if not os.path.exists(os.path.dirname(new_exps_path)) and os.path.exists(os.path.join("..", "data")):
    new_exps_path = os.path.join("..", new_exps_path)

if submit_btn:
    validation_passed = True
    errors = []
    
    if mpv > tpv:
        validation_passed = False
        errors.append("Physical Constraint Violated: Micropore Volume (MPV) cannot exceed Total Pore Volume (TPV).")
        
    comp_sum = c_perc + h_perc + n_perc + o_perc
    if not (80.0 <= comp_sum <= 105.0):
        validation_passed = False
        errors.append(f"Composition Balance Violated: The sum of C, H, N, and O must be approximately 100% (currently {comp_sum:.2f}%).")
        
    if not validation_passed:
        for err in errors:
            st.error(err)
    else:
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "SA": sa,
            "TPV": tpv,
            "MPV": mpv,
            "C": c_perc,
            "H": h_perc,
            "N": n_perc,
            "O": o_perc,
            "T": temp,
            "P": press,
            "CO2": co2_uptake,
            "Notes": notes
        }
        
        if os.path.exists(new_exps_path):
            try:
                new_exps_df = pd.read_csv(new_exps_path)
                new_exps_df = pd.concat([new_exps_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception:
                new_exps_df = pd.DataFrame([new_row])
        else:
            new_exps_df = pd.DataFrame([new_row])
            
        os.makedirs(os.path.dirname(new_exps_path), exist_ok=True)
        new_exps_df.to_csv(new_exps_path, index=False)
        st.success(f"Experimental result recorded successfully in {os.path.basename(new_exps_path)}!")
        st.balloons()
        st.rerun()

# ----------------- RETRAINING SECTION -----------------
st.write("")
st.markdown("### 🔄 Step 2: Active Learning Model Retraining")

if os.path.exists(new_exps_path):
    try:
        pending_df = pd.read_csv(new_exps_path)
        pending_count = len(pending_df)
    except Exception:
        pending_count = 0
else:
    pending_count = 0

st.write(f"Currently logged new experiments: **{pending_count}**")

if pending_count == 0:
    st.info("Log at least one experimental data point above to enable the retraining loop.")
else:
    retrain_btn = st.button("Retrain Models with New Data", type="secondary")
    
    if retrain_btn:
        with st.spinner("Executing retraining pipeline..."):
            try:
                result = retrain_models()
                
                if result:
                    st.success("Retraining pipeline executed successfully!")
                    
                    # Reload registry to show metrics
                    reg_after = load_model_registry()
                    old_v = result["old_version"]
                    new_v = result["new_version"]
                    
                    metrics_old = reg_after["versions"][old_v]["forward_metrics"]
                    metrics_new = reg_after["versions"][new_v]["forward_metrics"]
                    
                    col_metrics, col_status = st.columns([2, 1])
                    
                    with col_metrics:
                        st.markdown("##### 📊 Before/After Baseline Split Performance (Forward Model)")
                        metrics_df = pd.DataFrame({
                            "Metric": ["Test R²", "Test MSE", "Test RMSE", "Test MAPE"],
                            f"Active Version ({old_v})": [
                                f"{metrics_old['test_r2']:.4f}",
                                f"{metrics_old['test_mse']:.4f}",
                                f"{metrics_old['test_rmse']:.4f}",
                                f"{metrics_old['test_mape']:.2f}%"
                            ],
                            f"New Retrained ({new_v})": [
                                f"{metrics_new['test_r2']:.4f}",
                                f"{metrics_new['test_mse']:.4f}",
                                f"{metrics_new['test_rmse']:.4f}",
                                f"{metrics_new['test_mape']:.2f}%"
                            ]
                        })
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                    with col_status:
                        st.markdown("##### 📡 Active Status Report")
                        if result["activated"]:
                            st.markdown(
                                f"<div style='background-color:#065F46; padding: 1.25rem; border-radius: 8px; border: 1px solid #047857; text-align: center; height:100%'>"
                                f"<span style='color:#A7F3D0; font-weight:bold; font-size:1.2rem;'>ACTIVATED</span><br/>"
                                f"<span style='color:#E6FFFA; font-size:0.9rem;'>{new_v} is now live! R² improved on original test split.</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='background-color:#78350F; padding: 1.25rem; border-radius: 8px; border: 1px solid #D97706; text-align: center; height:100%'>"
                                f"<span style='color:#FDE68A; font-weight:bold; font-size:1.2rem;'>REJECTED</span><br/>"
                                f"<span style='color:#FFFDF5; font-size:0.9rem;'>{old_v} remains active. Performance did not improve.</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            st.warning(result["status_message"])
                else:
                    st.error("Retraining failed.")
            except Exception as e:
                st.error(f"Error during retraining: {str(e)}")
