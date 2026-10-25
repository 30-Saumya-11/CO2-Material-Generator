import os
import numpy as np
import pandas as pd

def generate_dataset():
    # Set random seed for reproducibility
    np.random.seed(42)
    n_rows = 573

    # SA (Surface Area): 2.5 - 3337 m²/g
    sa = np.random.exponential(scale=800, size=n_rows) + 2.5
    sa = np.clip(sa, 2.5, 3337.0)

    # TPV (Total Pore Volume): 0.02 - 3.09 cm³/g (correlated with SA)
    tpv = sa * 0.0009 + np.random.normal(0, 0.15, n_rows)
    tpv = np.clip(tpv, 0.02, 3.09)

    # MPV (Micropore Volume): 0.04 - 1.32 cm³/g (MPV <= TPV)
    mpv = tpv * np.random.uniform(0.4, 0.9, n_rows)
    mpv = np.clip(mpv, 0.04, 1.32)
    mpv = np.minimum(mpv, tpv)  # Physical constraint

    # Elemental composition (C, H, N, O)
    c_raw = np.random.uniform(60, 95, n_rows)
    h_raw = np.random.uniform(0.5, 5.0, n_rows)
    n_raw = np.random.uniform(0.1, 15.0, n_rows)
    o_raw = np.random.uniform(1.0, 25.0, n_rows)

    totals = c_raw + h_raw + n_raw + o_raw
    c = (c_raw / totals) * 98.0
    h = (h_raw / totals) * 98.0
    n = (n_raw / totals) * 98.0
    o = (o_raw / totals) * 98.0

    # Clip to specified ranges
    c = np.clip(c, 40.4, 96.96)
    h = np.clip(h, 0.0, 6.01)
    n = np.clip(n, 0.0, 38.15)
    o = np.clip(o, 0.0, 54.0)

    # Temperature (T): 0 - 30 °C (discrete: 0, 15, 25, 30)
    t = np.random.choice([0, 15, 25, 30], size=n_rows, p=[0.25, 0.25, 0.35, 0.15])

    # Pressure (P): 0.10 - 1.01 bar
    p = np.random.uniform(0.10, 1.01, size=n_rows)

    # CO2 uptake: 0.36 - 8.20 mmol/g
    # Based on a physically plausible relation: higher SA/MPV/N/P, lower T -> higher CO2
    co2 = (
        0.2 
        + 0.0012 * sa 
        + 1.8 * mpv 
        + 0.04 * n 
        - 0.03 * t 
        + 2.8 * p 
        + np.random.normal(0, 0.25, n_rows)
    )
    co2 = np.clip(co2, 0.36, 8.20)

    # Create DataFrame
    df = pd.DataFrame({
        'SA': sa,
        'TPV': tpv,
        'MPV': mpv,
        'C': c,
        'H': h,
        'N': n,
        'O': o,
        'T': t,
        'P': p,
        'CO2': co2
    })

    # Introduce ~8% missing values in TPV and MPV to validate preprocess.py imputation
    tpv_missing_idx = np.random.choice(n_rows, size=int(0.08 * n_rows), replace=False)
    mpv_missing_idx = np.random.choice(n_rows, size=int(0.08 * n_rows), replace=False)

    df.loc[tpv_missing_idx, 'TPV'] = np.nan
    df.loc[mpv_missing_idx, 'MPV'] = np.nan

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/raw_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Missing TPV count: {df['TPV'].isna().sum()}")
    print(f"Missing MPV count: {df['MPV'].isna().sum()}")

if __name__ == '__main__':
    generate_dataset()
