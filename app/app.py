import streamlit as st
import pandas as pd
import sys
import os

# Add src folder and workspace folder to PATH to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_config, load_model_registry

st.set_page_config(
    page_title="CO2 Material Generator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration and model registry
config = load_config()
registry = load_model_registry()

active_version = registry.get("active_version")
if active_version and active_version in registry.get("versions", {}):
    version_data = registry["versions"][active_version]
    fw_metrics = version_data["forward_metrics"]
    inv_metrics = version_data["inverse_metrics"]
    dataset_size = version_data["dataset_size"]
else:
    version_data = None
    fw_metrics = None
    inv_metrics = None
    dataset_size = 573 # default baseline

# Sidebar display
st.sidebar.markdown("### 🌱 Active ML Model Status")
st.sidebar.markdown(f"**Active Version:** `{active_version or 'N/A'}`")
st.sidebar.markdown(f"**Dataset Size:** `{dataset_size} materials`")

if fw_metrics:
    st.sidebar.markdown("#### Forward Model (GBDT)")
    st.sidebar.markdown(f"- **Test $R^2$:** `{fw_metrics['test_r2']:.4f}`")
    st.sidebar.markdown(f"- **Test MAPE:** `{fw_metrics['test_mape']:.2f}%`")
if inv_metrics:
    st.sidebar.markdown("#### Inverse Model (MultiOutput)")
    st.sidebar.markdown(f"- **Type:** `{inv_metrics['model_type'].upper()}`")
    st.sidebar.markdown(f"- **Val $R^2$:** `{inv_metrics['validation_r2']:.4f}`")

st.sidebar.info("Use the navigation pages in the sidebar to design porous carbons or upload new verification data.")

# Main Page Layout
st.markdown("<h1 class='main-title'>CO₂ Material Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Inverse ML System for Designing Porous Carbons for Carbon Capture Applications</p>", unsafe_allow_html=True)

# Grid Layout for Overview
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📖 Scientific Background")
    st.write(
        """
        This project is based on the research paper:
        **"Machine learning implemented exploration of the adsorption mechanism of carbon dioxide onto porous carbons"**
        *Journal of Colloid and Interface Science 647 (2023) 174–187*
        by Sarvesh Namdeo, Vimal Chandra Srivastava, and Paritosh Mohanty.
        
        Porous carbon materials are highly promising for carbon capture because they possess excellent chemical stability, 
        low synthesis cost, high surface areas, and customizable surface polarities. However, tuning synthesis parameters 
        (like surface area, micro/total pore volumes, and elemental concentration) to reach a target CO₂ adsorption level 
        is a highly complex, multi-variable challenge.
        
        This system implements:
        1. **A Forward Prediction Model:** Predicts CO₂ adsorption (mmol/g) given material parameters and environmental conditions (T, P).
        2. **An Inverse Material Generator:** Generates synthesis candidates (property profiles) to meet a specified target CO₂ uptake.
        3. **An Active Learning Retraining Loop:** Seamlessly incorporates new experimental data and retrains/updates the models.
        """
    )
    
    st.markdown("### 🚀 Quick Guide")
    st.markdown(
        """
        * **1. Material Generator (`01_generate`):** Input a target CO₂ value, lock specific Temperature/Pressure parameters if needed, and view the top 3 material candidates, along with verification scores, warning flags, and feature contributions via SHAP charts.
        * **2. Add New Experiment (`02_add_experiment`):** Enter experimental lab results to log new materials. Retrain the model on the fly and monitor if the new version becomes active.
        * **3. Model History (`03_model_history`):** Visualize how metrics improve as the dataset expands, check the calibration curves, and roll back models if performance drops.
        """
    )

with col2:
    st.markdown("### 📊 Material Feature Glossary")
    st.markdown(
        """
        The models utilize 9 structural and elemental characteristics:
        
        * **SA:** Surface Area ($m^2/g$)
        * **TPV:** Total Pore Volume ($cm^3/g$)
        * **MPV:** Micropore Volume ($cm^3/g$)
        * **C:** Carbon composition (%)
        * **H:** Hydrogen composition (%)
        * **N:** Nitrogen composition (%)
        * **O:** Oxygen composition (%)
        * **T:** Temperature of operation (°C)
        * **P:** Pressure of operation ($bar$)
        
        *Target Output:* **CO2** uptake capacity ($mmol/g$)
        """
    )

# Training Bounds Table
st.markdown("### 📈 Training bounds reference (original dataset boundaries)")
features_info = []
feature_ranges = config['feature_ranges']
for feat, limits in feature_ranges.items():
    features_info.append({
        "Feature Name": feat,
        "Minimum Limit": f"{limits['min']:.2f}",
        "Maximum Limit": f"{limits['max']:.2f}",
        "Unit": limits.get('unit', '')
    })
df_bounds = pd.DataFrame(features_info)
st.dataframe(df_bounds, use_container_width=True, hide_index=True)
