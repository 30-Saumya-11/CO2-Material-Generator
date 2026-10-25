# CO2 Material Generator

An end-to-end Inverse Machine Learning (ML) system designed to recommend porous carbon material properties needed to achieve a target CO2 adsorption uptake (in mmol/g).

This project is inspired by the research paper:
> "Machine learning implemented exploration of the adsorption mechanism of carbon dioxide onto porous carbons"
> Sarvesh Namdeo, Vimal Chandra Srivastava, Paritosh Mohanty
> Published in Journal of Colloid and Interface Science 647 (2023) 174-187

---

## Dataset Description and Bounds

> **Data note:** the dataset in this public repository is **synthetic**. `scripts/generate_raw_data.py` generates 573 rows within the feature ranges reported in the paper (seed 42), so the pipeline can be run end to end. Metrics computed on it describe the pipeline, not real materials.

The system is built on a dataset consisting of 573 baseline materials with the following structural and environmental features:

| Feature Name | Description | Unit | Range |
| :--- | :--- | :--- | :--- |
| SA | Specific Surface Area | m2/g | 2.5 - 3337 |
| TPV | Total Pore Volume | cm3/g | 0.02 - 3.09 |
| MPV | Micropore Volume | cm3/g | 0.04 - 1.32 |
| C | Carbon Content | % | 40.4 - 96.96 |
| H | Hydrogen Content | % | 0 - 6.01 |
| N | Nitrogen Content | % | 0 - 38.15 |
| O | Oxygen Content | % | 0 - 54 |
| T | Temperature | C | 0 - 30 |
| P | Pressure | bar | 0.10 - 1.01 |
| CO2 (Target) | CO2 adsorption uptake | mmol/g | 0.36 - 8.20 |

---

## Project Structure

```
co2_material_generator/
|
|-- data/
|   |-- raw_dataset.csv          # 573 baseline rows (with some missing TPV/MPV values)
|   |-- processed_dataset.csv    # Cleaned, imputed, and scaled features
|   |-- new_experiments.csv      # Logged user experiments
|   └-- combined_dataset.csv     # Merged dataset used in retraining
|
|-- models/
|   |-- forward_model_v1.pkl     # GBDT forward predictor
|   |-- forward_explainer_v1.pkl # SHAP TreeExplainer object
|   |-- inverse_model_v1.pkl     # Selected MultiOutput inverse model (GBDT or RF)
|   |-- scaler_v1.pkl            # StandardScaler fitted on v1 inputs
|   └-- model_registry.json      # Metadata tracking version histories and metrics
|
|-- src/
|   |-- preprocess.py            # Loads raw, imputes missing TPV/MPV with RF, scales data
|   |-- train_forward.py         # Trains and registers forward GBDT model + SHAP
|   |-- train_inverse.py         # Trains and compares MultiOutput wrapper regressors
|   |-- confidence.py            # Safety boundary analyzer and score calculator
|   |-- generate_material.py     # Multi-step perturbation and candidate selection pipeline
|   |-- retrain.py               # Incorporates new data, validates metrics, handles deployment
|   └-- utils.py                 # Common helper utilities for registry, loading, saving, and errors
|
|-- app/
|   |-- app.py                   # Streamlit landing page with metadata
|   └-- pages/
|       |-- 01_generate.py       # Candidate generator UI and SHAP bar charts
|       |-- 02_add_experiment.py # Laboratory experiment log form and retrain trigger
|       └-- 03_model_history.py  # Model history, progression charts, and rollback portal
|
|-- config.yaml                  # Constants, physical ranges, thresholds, and paths
|-- requirements.txt             # Python environment dependencies
└-- README.md                    # Documentation
```

---

## Core Architecture and Pipelines

### 1. Missing Value Imputation

The preprocessing pipeline (src/preprocess.py) automatically handles missing values using a Random Forest approach:
- TPV is imputed using a Random Forest regressor trained on SA + MPV.
- MPV is imputed using a Random Forest regressor trained on SA + TPV.
- A physical constraint is enforced post-imputation: MPV <= TPV.

### 2. Forward Predictor (Inputs -> CO2)

Trains a Gradient Boosting Regressor (GBDT) with hyperparameters:
- n_estimators = 200, learning_rate = 0.05, max_depth = 4.
- Outputs R2, MSE, RMSE, and MAPE. A SHAP explainer object is saved alongside for explaining local feature attributions.

### 3. Inverse Model (CO2 -> Inputs)

Solves a multi-output regression problem. The script (src/train_inverse.py):
- Wraps both GradientBoostingRegressor and RandomForestRegressor in a MultiOutputRegressor.
- Evaluates them using a Validation R2 (where recommended properties are passed into the forward model to see how close the predicted CO2 is to the target).
- Deploys the superior wrapper to models/.

### 4. Generation Pipeline and Perturbations

When a target CO2 is inputted (src/generate_material.py):
1. Confidence Analysis: Flags target values as SAFE (within training range [0.36 - 8.20]), HYPOTHESIS (within 20% extrapolation), or SPECULATIVE (> 20% extrapolation).
2. Inverse Recommendation: Generates baseline properties.
3. Synthesis Lock Overage: Overrides Temperature or Pressure if the user requested locked constraints.
4. Perturbations: Adds small +/- 5% random noise to create 10 candidate variations, while enforcing physical constraints (MPV <= TPV, composition sum ~100%).
5. Forward Verification: Passes all 10 candidates through the forward model.
6. Ranking: Ranks candidates by proximity of predicted validation CO2 to the target CO2 and returns the top 3.

### 5. Active Learning Retraining Loop

When a user adds new verified lab experiments (src/retrain.py):
- Appends new experiments to the raw dataset and fits a new scaler_vN.pkl.
- Retrains forward and inverse models.
- Fair Baseline Check: Evaluates the retrained model and old model on the original 15% test set to see if R2 has improved.
- If R2 improves: Actively promotes version vN to live.
- If R2 degrades: Retains the old model version as live and logs/displays a warning.

---

## How to Run

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Generate Dataset and Run Baseline Training

Initialize the raw database, preprocess it, and train the baseline models (v1) sequentially:

```bash
# Create raw dataset
python3 scripts/generate_raw_data.py

# Run preprocess and imputation
python3 src/preprocess.py

# Train forward model and save SHAP explainer
python3 src/train_forward.py

# Train inverse model and register metrics
python3 src/train_inverse.py
```

### 3. Launch the Streamlit App

```bash
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.
