import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
from datetime import datetime
from utils import load_config, save_model, load_model_registry, save_model_registry, compute_mape

def train_forward_model(processed_path=None, version="v1"):
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
    
    input_cols = config['columns'][:-1]
    target_col = config['columns'][-1]
    
    X = df[input_cols]
    y = df[target_col]
    
    # Train-test split (85% train, 15% test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("Training GradientBoostingRegressor forward model...")
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mape = compute_mape(y_train, y_train_pred)
    test_mape = compute_mape(y_test, y_test_pred)
    
    print("\nForward Model Evaluation Metrics:")
    print(f"Train R2:   {train_r2:.4f}")
    print(f"Test R2:    {test_r2:.4f}")
    print(f"Train MSE:  {train_mse:.4f}")
    print(f"Test MSE:   {test_mse:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Test MAPE:  {test_mape:.2f}%")
    
    # Save the model
    save_model(model, "forward_model", version)
    
    # Save SHAP Explainer
    print("Generating and saving SHAP TreeExplainer...")
    # TreeExplainer is fast and native to sklearn GBDT
    explainer = shap.TreeExplainer(model)
    save_model(explainer, "forward_explainer", version)
    
    # Update registry
    registry = load_model_registry()
    if "versions" not in registry:
        registry["versions"] = {}
        
    # We maintain the active version or update it
    registry["versions"][version] = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(df),
        "forward_metrics": {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_mape": float(train_mape),
            "test_mape": float(test_mape)
        },
        "inverse_metrics": {}  # To be filled by train_inverse.py
    }
    
    if registry.get("active_version") is None:
        registry["active_version"] = version
        
    save_model_registry(registry)
    print("Model registered in registry.")

if __name__ == '__main__':
    train_forward_model()
