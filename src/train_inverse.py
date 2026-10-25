import os
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import load_config, save_model, load_model, load_model_registry, save_model_registry

def train_inverse_model(processed_path=None, version="v1"):
    config = load_config()
    if processed_path is None:
        processed_path = config['paths']['processed_dataset']
        
    print(f"Loading processed dataset from {processed_path}...")
    if not os.path.exists(processed_path):
        if os.path.exists(os.path.join("..", processed_path)):
            processed_path = os.path.join("..", processed_path)
        else:
            raise FileNotFoundError(f"Processed dataset not found at {processed_path}")
            
    df = pd.read_csv(processed_path)
    
    input_cols = config['columns'][:-1] # SA, TPV, MPV, C, H, N, O, T, P
    target_col = config['columns'][-1]  # CO2
    
    X_inv = df[[target_col]]
    y_inv = df[input_cols]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_inv, y_inv, test_size=0.15, random_state=42)
    
    # Train GBDT wrapper
    print("Training MultiOutputRegressor wrapping GradientBoostingRegressor...")
    model_gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
    model_gb.fit(X_train, y_train)
    
    # Train RF wrapper
    print("Training MultiOutputRegressor wrapping RandomForestRegressor...")
    model_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model_rf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred_gb = model_gb.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    
    # Compute R2 for each feature
    r2_gb_list = []
    r2_rf_list = []
    
    for i, col in enumerate(input_cols):
        r2_gb = r2_score(y_test.iloc[:, i], y_pred_gb[:, i])
        r2_rf = r2_score(y_test.iloc[:, i], y_pred_rf[:, i])
        r2_gb_list.append(r2_gb)
        r2_rf_list.append(r2_rf)
        
    avg_r2_gb = np.mean(r2_gb_list)
    avg_r2_rf = np.mean(r2_rf_list)
    
    print("\nFeature-level R2 scores on test set:")
    print(f"{'Feature':<10} | {'GBDT Wrapper':<15} | {'RF Wrapper':<15}")
    print("-" * 50)
    for i, col in enumerate(input_cols):
        print(f"{col:<10} | {r2_gb_list[i]:.4f}          | {r2_rf_list[i]:.4f}")
    print("-" * 50)
    print(f"{'Average R2':<10} | {avg_r2_gb:.4f}          | {avg_r2_rf:.4f}")
    
    # Load forward model to compute validation R2
    forward_model = load_model("forward_model", version)
    
    # Predict and evaluate validation R2
    val_co2_pred_gb = forward_model.predict(y_pred_gb)
    val_co2_pred_rf = forward_model.predict(y_pred_rf)
    
    val_r2_gb = r2_score(X_test[target_col], val_co2_pred_gb)
    val_r2_rf = r2_score(X_test[target_col], val_co2_pred_rf)
    
    print(f"\nValidation R2 (inverse predicted features -> forward model -> predicted CO2 vs actual CO2):")
    print(f"GBDT Wrapper: {val_r2_gb:.4f}")
    print(f"RF Wrapper:   {val_r2_rf:.4f}")
    
    # Select better model based on validation R2
    if val_r2_gb >= val_r2_rf:
        print("\nSelecting Gradient Boosting MultiOutput model as the active inverse model.")
        best_model = model_gb
        best_type = "gbdt"
        best_r2_list = r2_gb_list
        best_val_r2 = val_r2_gb
    else:
        print("\nSelecting Random Forest MultiOutput model as the active inverse model.")
        best_model = model_rf
        best_type = "random_forest"
        best_r2_list = r2_rf_list
        best_val_r2 = val_r2_rf
        
    # Save the selected inverse model
    save_model(best_model, "inverse_model", version)
    
    # Update the model registry
    registry = load_model_registry()
    if "versions" not in registry:
        registry["versions"] = {}
        
    if version not in registry["versions"]:
        registry["versions"][version] = {}
        
    feature_r2_dict = {col: float(r) for col, r in zip(input_cols, best_r2_list)}
    
    registry["versions"][version]["inverse_metrics"] = {
        "model_type": best_type,
        "validation_r2": float(best_val_r2),
        "feature_r2": feature_r2_dict
    }
    
    save_model_registry(registry)
    print("Inverse model registered in registry.")

if __name__ == '__main__':
    train_inverse_model()
