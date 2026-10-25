import os
import yaml
import joblib
import json
import numpy as np

def load_config(config_path="config.yaml"):
    """Loads configuration details from config.yaml."""
    if os.path.exists(config_path):
        target = config_path
    elif os.path.exists(os.path.join("..", config_path)):
        target = os.path.join("..", config_path)
    else:
        target = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        
    if not os.path.exists(target):
        raise FileNotFoundError(f"Could not locate {config_path}")
        
    with open(target, 'r') as f:
        return yaml.safe_load(f)

def load_model_registry(registry_path="models/model_registry.json"):
    """Loads the model registry JSON file."""
    if os.path.exists(registry_path):
        target = registry_path
    elif os.path.exists(os.path.join("..", registry_path)):
        target = os.path.join("..", registry_path)
    else:
        target = os.path.join(os.path.dirname(__file__), "..", "models", "model_registry.json")
        
    if not os.path.exists(target):
        # Return a default structure
        return {"active_version": None, "versions": {}}
            
    with open(target, 'r') as f:
        return json.load(f)

def save_model_registry(registry, registry_path="models/model_registry.json"):
    """Saves the model registry JSON file atomically."""
    if os.path.exists(os.path.dirname(registry_path)):
        target_dir = os.path.dirname(registry_path)
        target = registry_path
    elif os.path.exists(os.path.join("..", "models")):
        target_dir = os.path.join("..", "models")
        target = os.path.join(target_dir, "model_registry.json")
    else:
        target_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        target = os.path.join(target_dir, "model_registry.json")
        
    os.makedirs(target_dir, exist_ok=True)
    temp_path = target + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(registry, f, indent=4)
    os.replace(temp_path, target)

def save_model(model, name, version, models_dir="models"):
    """Saves a model object using joblib."""
    if os.path.exists(models_dir):
        target_dir = models_dir
    elif os.path.exists(os.path.join("..", models_dir)):
        target_dir = os.path.join("..", models_dir)
    else:
        target_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, f"{name}_{version}.pkl")
    joblib.dump(model, filepath)
    print(f"Saved model to {filepath}")

def load_model(name, version, models_dir="models"):
    """Loads a model object using joblib."""
    if os.path.exists(models_dir):
        target_dir = models_dir
    elif os.path.exists(os.path.join("..", models_dir)):
        target_dir = os.path.join("..", models_dir)
    else:
        target_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        
    filepath = os.path.join(target_dir, f"{name}_{version}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")
    return joblib.load(filepath)

def compute_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def inverse_transform_features(X_scaled, scaler, feature_cols=None):
    """
    Applies inverse transformation to scaled features.
    If feature_cols is provided, returns a pandas DataFrame with correct columns.
    """
    X_inv = scaler.inverse_transform(X_scaled)
    if feature_cols is not None:
        import pandas as pd
        return pd.DataFrame(X_inv, columns=feature_cols)
    return X_inv
