import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from utils import load_config, save_model

def preprocess_data(raw_path=None, processed_path=None, version="v1"):
    config = load_config()
    
    if raw_path is None:
        raw_path = config['paths']['raw_dataset']
    if processed_path is None:
        processed_path = config['paths']['processed_dataset']
        
    print(f"Loading raw dataset from {raw_path}...")
    if not os.path.exists(raw_path):
        # Fallback to parent path
        if os.path.exists(os.path.join("..", raw_path)):
            raw_path = os.path.join("..", raw_path)
        else:
            raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
            
    df = pd.read_csv(raw_path)
    
    # Standardize column names (strip spaces, uppercase)
    col_mapping = {c: c.strip().upper() for c in df.columns}
    # Special handle for CO2
    for k, v in col_mapping.items():
        if v == 'CO2' or 'CO2' in v or 'CO₂' in v:
            col_mapping[k] = 'CO2'
    df = df.rename(columns=col_mapping)
    
    # Ensure all required columns are present
    req_cols = config['columns']
    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} missing from dataset!")
            
    df = df[req_cols].copy()
    
    # Handle missing values using RF as described in the paper
    tpv_missing = df['TPV'].isna()
    mpv_missing = df['MPV'].isna()
    
    sa_median = df['SA'].median()
    tpv_median = df['TPV'].median()
    mpv_median = df['MPV'].median()
    
    # Ensure SA is filled
    df['SA'] = df['SA'].fillna(sa_median)
    
    # Train RF on complete rows for TPV and MPV
    complete_idx = (~df['TPV'].isna()) & (~df['MPV'].isna())
    complete_data = df[complete_idx]
    
    if len(complete_data) < 10:
        # Fallback: if not enough complete data, fill with medians
        print("Warning: Not enough complete data for RF imputation. Falling back to median imputation.")
        df['TPV'] = df['TPV'].fillna(tpv_median)
        df['MPV'] = df['MPV'].fillna(mpv_median)
    else:
        if tpv_missing.any():
            print(f"Imputing {tpv_missing.sum()} missing TPV values...")
            rf_tpv = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_tpv.fit(complete_data[['SA', 'MPV']], complete_data['TPV'])
            
            missing_tpv_df = df[tpv_missing].copy()
            temp_mpv = missing_tpv_df['MPV'].fillna(mpv_median)
            imputed_tpv = rf_tpv.predict(pd.DataFrame({'SA': missing_tpv_df['SA'], 'MPV': temp_mpv}))
            df.loc[tpv_missing, 'TPV'] = imputed_tpv
            
        if mpv_missing.any():
            print(f"Imputing {mpv_missing.sum()} missing MPV values...")
            rf_mpv = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_mpv.fit(complete_data[['SA', 'TPV']], complete_data['MPV'])
            
            missing_mpv_df = df[mpv_missing].copy()
            temp_tpv = df.loc[mpv_missing, 'TPV'].fillna(tpv_median)
            imputed_mpv = rf_mpv.predict(pd.DataFrame({'SA': missing_mpv_df['SA'], 'TPV': temp_tpv}))
            df.loc[mpv_missing, 'MPV'] = imputed_mpv
            
    # Enforce physical constraints: MPV <= TPV
    over_mpv = df['MPV'] > df['TPV']
    if over_mpv.any():
        print(f"Enforcing MPV <= TPV constraint on {over_mpv.sum()} rows...")
        df.loc[over_mpv, 'MPV'] = df.loc[over_mpv, 'TPV']
        
    # Scale inputs
    input_cols = req_cols[:-1]  # SA, TPV, MPV, C, H, N, O, T, P
    target_col = req_cols[-1]   # CO2
    
    X = df[input_cols].values
    y = df[target_col].values
    
    print("Fitting StandardScaler on input features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_processed = pd.DataFrame(X_scaled, columns=input_cols)
    df_processed[target_col] = y
    
    # Determine the directory to write processed dataset
    proc_dir = os.path.dirname(processed_path)
    if not proc_dir:
        proc_dir = os.path.dirname(os.path.join("..", processed_path))
    os.makedirs(proc_dir, exist_ok=True)
    
    df_processed.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to {processed_path}")
    
    # Save the scaler
    save_model(scaler, "scaler", version)
    
    print("\nProcessed Dataset Summary Statistics (Original Units):")
    print(df.describe().to_string())
    
    return df_processed, scaler

if __name__ == '__main__':
    preprocess_data()
