import numpy as np
import pandas as pd
from utils import load_config, load_model, load_model_registry, inverse_transform_features
from confidence import evaluate_material_confidence

def generate_candidates(co2_target, fixed_temp=None, fixed_press=None, version=None):
    """
    Takes target CO2 value and optional temperature/pressure constraints,
    recommends porous carbon properties, creates perturbations, validates, and ranks.
    Returns: list of top 3 candidates.
    """
    config = load_config()
    registry = load_model_registry()
    
    if version is None:
        version = registry.get("active_version", "v1")
        
    # Load forward/inverse models and scaler
    forward_model = load_model("forward_model", version)
    inverse_model = load_model("inverse_model", version)
    scaler = load_model("scaler", version)
    
    input_cols = config['columns'][:-1]
    
    # 1. Feed target CO2 into inverse model
    X_inv_input = pd.DataFrame({'CO2': [co2_target]})
    scaled_pred = inverse_model.predict(X_inv_input)
    
    # 2. Convert to original units
    pred_orig = inverse_transform_features(scaled_pred, scaler, input_cols)
    base_properties = pred_orig.iloc[0].to_dict()
    
    # Override fixed temperature or pressure constraints if provided
    if fixed_temp is not None:
        base_properties['T'] = float(fixed_temp)
    if fixed_press is not None:
        base_properties['P'] = float(fixed_press)
        
    # Enforce physical constraints: MPV <= TPV
    base_properties['MPV'] = min(base_properties['MPV'], base_properties['TPV'])
    
    # 3. Create 10 variants (1 base + 9 perturbed)
    variants = []
    variants.append(base_properties.copy())  # Base candidate has 0% perturbation
    
    np.random.seed(42)  # For reproducible perturbations
    feature_ranges = config['feature_ranges']
    
    for _ in range(9):
        variant = {}
        for feat in input_cols:
            val = base_properties[feat]
            
            # If the feature is locked, don't perturb it
            if feat == 'T' and fixed_temp is not None:
                variant['T'] = float(fixed_temp)
                continue
            if feat == 'P' and fixed_press is not None:
                variant['P'] = float(fixed_press)
                continue
                
            # Perturb by ±5%
            perturb = np.random.uniform(-0.05, 0.05)
            val_perturbed = val * (1.0 + perturb)
            
            # Clip within physical range
            limits = feature_ranges[feat]
            val_perturbed = np.clip(val_perturbed, limits['min'], limits['max'])
            variant[feat] = float(val_perturbed)
            
            # Additional constraint: discrete temperature rounding if fixed isn't specified
            # Usually T in the dataset is discrete (0, 15, 25, 30).
            # If not fixed, round T to nearest integer or keep as float. Let's keep it as float for general ML flexibility.
            
        # Enforce structural/composition constraints
        variant['MPV'] = min(variant['MPV'], variant['TPV'])
        
        # Composition normalization: sum(C, H, N, O) should match base sum
        comp_cols = ['C', 'H', 'N', 'O']
        base_sum = sum(base_properties[c] for c in comp_cols)
        var_sum = sum(variant[c] for c in comp_cols)
        if var_sum > 0:
            for c in comp_cols:
                variant[c] = (variant[c] / var_sum) * base_sum
                
        # Clip composition features to training limits
        for c in comp_cols:
            limits = feature_ranges[c]
            variant[c] = np.clip(variant[c], limits['min'], limits['max'])
            
        variants.append(variant)
        
    # Convert variants to DataFrame
    df_variants = pd.DataFrame(variants)
    
    # 4. Transform variants back to scaled space
    X_scaled = scaler.transform(df_variants[input_cols].values)
    
    # 5. Feed into forward model
    predicted_co2 = forward_model.predict(X_scaled)
    df_variants['Validation_CO2'] = predicted_co2
    
    # 6. Rank by closeness to target CO2
    df_variants['Error'] = np.abs(df_variants['Validation_CO2'] - co2_target)
    df_variants = df_variants.sort_values(by='Error').reset_index(drop=True)
    
    # 7. Package top 3 candidates
    top_candidates = []
    for i in range(min(3, len(df_variants))):
        cand_properties = df_variants.loc[i, input_cols].to_dict()
        val_co2 = df_variants.loc[i, 'Validation_CO2']
        
        # Run confidence report
        conf = evaluate_material_confidence(co2_target, cand_properties, val_co2, config)
        
        top_candidates.append({
            "properties": cand_properties,
            "validation_co2": float(val_co2),
            "validation_error_pct": conf["validation_error_pct"],
            "confidence_score": conf["overall_score"],
            "confidence_level": conf["level"],
            "confidence_details": conf
        })
        
    return top_candidates

if __name__ == '__main__':
    # Quick CLI test
    import sys
    co2_val = 5.5
    if len(sys.argv) > 1:
        co2_val = float(sys.argv[1])
    try:
        candidates = generate_candidates(co2_val)
        print(f"\nTop Candidates for CO2 = {co2_val} mmol/g:")
        for idx, c in enumerate(candidates):
            print(f"\nCandidate {idx+1}:")
            print(f"  Confidence Level: {c['confidence_level']} ({c['confidence_score']:.1f}%)")
            print(f"  Validation CO2:   {c['validation_co2']:.3f} (Error: {c['validation_error_pct']:.2f}%)")
            print(f"  Properties:       { {k: round(v, 2) for k, v in c['properties'].items()} }")
    except Exception as e:
        print("Could not run generation: models not trained yet.", e)
