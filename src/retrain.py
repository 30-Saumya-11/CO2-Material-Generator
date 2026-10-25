import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
from utils import load_config, save_model, load_model, load_model_registry, save_model_registry, compute_mape

def retrain_models():
    """
    Appends new experiments, merges, retrains models, compares R2 on baseline test set,
    and updates active model if it improves.
    """
    config = load_config()
    registry = load_model_registry()
    
    active_version = registry.get("active_version")
    if active_version is None:
        raise ValueError("No active model version found in registry. Please train v1 first.")
        
    new_exps_path = config['paths']['new_experiments']
    if not os.path.exists(new_exps_path):
        if os.path.exists(os.path.join("..", new_exps_path)):
            new_exps_path = os.path.join("..", new_exps_path)
        else:
            print("No new experiments found. Retraining skipped.")
            return False
            
    print(f"Loading new experiments from {new_exps_path}...")
    new_df = pd.read_csv(new_exps_path)
    if len(new_df) == 0:
        print("New experiments file is empty. Retraining skipped.")
        return False
        
    # Standardize column names
    col_mapping = {c: c.strip().upper() for c in new_df.columns}
    for k, v in col_mapping.items():
        if v == 'CO2' or 'CO2' in v or 'CO₂' in v:
            col_mapping[k] = 'CO2'
    new_df = new_df.rename(columns=col_mapping)
    
    req_cols = config['columns']
    for col in req_cols:
        if col not in new_df.columns:
            raise ValueError(f"Required column {col} missing from new experiments!")
            
    new_df = new_df[req_cols].copy()
    
    # Load raw baseline data
    raw_path = config['paths']['raw_dataset']
    if not os.path.exists(raw_path) and os.path.exists(os.path.join("..", raw_path)):
        raw_path = os.path.join("..", raw_path)
        
    print(f"Loading original raw dataset from {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    
    # Standardize raw_df columns
    col_mapping_raw = {c: c.strip().upper() for c in raw_df.columns}
    for k, v in col_mapping_raw.items():
        if v == 'CO2' or 'CO2' in v or 'CO₂' in v:
            col_mapping_raw[k] = 'CO2'
    raw_df = raw_df.rename(columns=col_mapping_raw)
    raw_df = raw_df[req_cols].copy()
    
    # Impute missing values on baseline dataset first
    # This keeps original 573 baseline imputation consistent before adding new rows
    tpv_missing = raw_df['TPV'].isna()
    mpv_missing = raw_df['MPV'].isna()
    
    sa_median = raw_df['SA'].median()
    tpv_median = raw_df['TPV'].median()
    mpv_median = raw_df['MPV'].median()
    
    raw_df['SA'] = raw_df['SA'].fillna(sa_median)
    complete_idx = (~raw_df['TPV'].isna()) & (~raw_df['MPV'].isna())
    complete_data = raw_df[complete_idx]
    
    if len(complete_data) >= 10:
        if tpv_missing.any():
            rf_tpv = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_tpv.fit(complete_data[['SA', 'MPV']], complete_data['TPV'])
            missing_tpv_df = raw_df[tpv_missing].copy()
            temp_mpv = missing_tpv_df['MPV'].fillna(mpv_median)
            imputed_tpv = rf_tpv.predict(pd.DataFrame({'SA': missing_tpv_df['SA'], 'MPV': temp_mpv}))
            raw_df.loc[tpv_missing, 'TPV'] = imputed_tpv
            
        if mpv_missing.any():
            rf_mpv = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_mpv.fit(complete_data[['SA', 'TPV']], complete_data['MPV'])
            missing_mpv_df = raw_df[mpv_missing].copy()
            temp_tpv = raw_df.loc[mpv_missing, 'TPV'].fillna(tpv_median)
            imputed_mpv = rf_mpv.predict(pd.DataFrame({'SA': missing_mpv_df['SA'], 'TPV': temp_tpv}))
            raw_df.loc[mpv_missing, 'MPV'] = imputed_mpv
    else:
        raw_df['TPV'] = raw_df['TPV'].fillna(tpv_median)
        raw_df['MPV'] = raw_df['MPV'].fillna(mpv_median)
        
    raw_df['MPV'] = np.minimum(raw_df['MPV'], raw_df['TPV'])
    
    # Split clean baseline into train/test (85% / 15%, random_state=42)
    # The 15% test set MUST be kept separate and not used in new training to ensure fair evaluation
    base_train, base_test = train_test_split(raw_df, test_size=0.15, random_state=42)
    
    # Merge training data: baseline train + new experiments
    combined_train = pd.concat([base_train, new_df], ignore_index=True)
    
    input_cols = req_cols[:-1]
    target_col = req_cols[-1]
    
    # Fit new StandardScaler on the combined training set
    print("Fitting new StandardScaler on combined training set...")
    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(combined_train[input_cols].values)
    y_train = combined_train[target_col].values
    
    # Scale the original baseline test set
    X_test_scaled_new = new_scaler.transform(base_test[input_cols].values)
    y_test = base_test[target_col].values
    
    # Versioning
    version_num = int(active_version.replace("v", ""))
    new_version = f"v{version_num + 1}"
    print(f"Training models for version {new_version}...")
    
    # Retrain GBDT Forward Model
    new_forward = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    new_forward.fit(X_train_scaled, y_train)
    
    # Evaluate new forward model on original test set
    y_test_pred_new = new_forward.predict(X_test_scaled_new)
    new_r2 = r2_score(y_test, y_test_pred_new)
    new_mse = mean_squared_error(y_test, y_test_pred_new)
    new_rmse = np.sqrt(new_mse)
    new_mape = compute_mape(y_test, y_test_pred_new)
    
    # Load old forward model and scaler to compute baseline test performance for comparison
    old_forward = load_model("forward_model", active_version)
    old_scaler = load_model("scaler", active_version)
    X_test_scaled_old = old_scaler.transform(base_test[input_cols].values)
    y_test_pred_old = old_forward.predict(X_test_scaled_old)
    
    old_r2 = r2_score(y_test, y_test_pred_old)
    old_mse = mean_squared_error(y_test, y_test_pred_old)
    old_rmse = np.sqrt(old_mse)
    old_mape = compute_mape(y_test, y_test_pred_old)
    
    print("\nForward Model Comparison on Original Test Set:")
    print(f"{'Metric':<10} | {active_version:<10} | {new_version:<10}")
    print("-" * 35)
    print(f"{'R2':<10} | {old_r2:.4f}     | {new_r2:.4f}")
    print(f"{'MSE':<10} | {old_mse:.4f}     | {new_mse:.4f}")
    print(f"{'RMSE':<10} | {old_rmse:.4f}     | {new_rmse:.4f}")
    print(f"{'MAPE':<10} | {old_mape:.2f}%     | {new_mape:.2f}%")
    
    # Train MultiOutput inverse models on new training set
    X_inv_train = combined_train[[target_col]]
    y_inv_train = combined_train[input_cols]
    
    X_inv_test = base_test[[target_col]]
    y_inv_test = base_test[input_cols]
    
    # GBDT
    inv_gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
    inv_gb.fit(X_inv_train, y_inv_train)
    
    # RF
    inv_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    inv_rf.fit(X_inv_train, y_inv_train)
    
    # Evaluate validation R2 on test set using the new forward model
    y_pred_gb = inv_gb.predict(X_inv_test)
    y_pred_rf = inv_rf.predict(X_inv_test)
    
    val_co2_gb = new_forward.predict(y_pred_gb)
    val_co2_pred_rf = new_forward.predict(y_pred_rf)
    
    val_r2_gb = r2_score(y_test, val_co2_gb)
    val_r2_rf = r2_score(y_test, val_co2_pred_rf)
    
    if val_r2_gb >= val_r2_rf:
        new_inverse = inv_gb
        best_inv_type = "gbdt"
        best_val_r2 = val_r2_gb
        best_inv_preds = y_pred_gb
    else:
        new_inverse = inv_rf
        best_inv_type = "random_forest"
        best_val_r2 = val_r2_rf
        best_inv_preds = y_pred_rf
        
    feature_r2_dict = {}
    for i, col in enumerate(input_cols):
        r_val = r2_score(y_inv_test.iloc[:, i], best_inv_preds[:, i])
        feature_r2_dict[col] = float(r_val)
        
    # Save the new version artifacts
    save_model(new_forward, "forward_model", new_version)
    save_model(new_inverse, "inverse_model", new_version)
    save_model(new_scaler, "scaler", new_version)
    
    new_explainer = shap.TreeExplainer(new_forward)
    save_model(new_explainer, "forward_explainer", new_version)
    
    # Save the combined raw dataset
    combined_raw_path = config['paths']['combined_dataset']
    if not os.path.exists(os.path.dirname(combined_raw_path)) and os.path.exists(os.path.join("..", "data")):
        combined_raw_path = os.path.join("..", combined_raw_path)
    combined_raw = pd.concat([raw_df, new_df], ignore_index=True)
    combined_raw.to_csv(combined_raw_path, index=False)
    
    # Determine activation logic: R2 must not decrease
    activate_new = new_r2 >= old_r2
    
    registry["versions"][new_version] = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(combined_raw),
        "forward_metrics": {
            "train_r2": float(r2_score(y_train, new_forward.predict(X_train_scaled))),
            "test_r2": float(new_r2),
            "train_mse": float(mean_squared_error(y_train, new_forward.predict(X_train_scaled))),
            "test_mse": float(new_mse),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, new_forward.predict(X_train_scaled)))),
            "test_rmse": float(new_rmse),
            "train_mape": float(compute_mape(y_train, new_forward.predict(X_train_scaled))),
            "test_mape": float(new_mape)
        },
        "inverse_metrics": {
            "model_type": best_inv_type,
            "validation_r2": float(best_val_r2),
            "feature_r2": feature_r2_dict
        }
    }
    
    status_msg = ""
    if activate_new:
        registry["active_version"] = new_version
        status_msg = f"Version {new_version} activated successfully! Test R2 improved from {old_r2:.4f} to {new_r2:.4f}."
    else:
        status_msg = f"WARNING: Version {new_version} trained but NOT activated. Performance degraded from R2 = {old_r2:.4f} to {new_r2:.4f}. Active version remains {active_version}."
        
    save_model_registry(registry)
    print("\n" + status_msg)
    
    return {
        "success": True,
        "activated": activate_new,
        "old_version": active_version,
        "new_version": new_version,
        "old_r2": old_r2,
        "new_r2": new_r2,
        "status_message": status_msg
    }

if __name__ == '__main__':
    retrain_models()
