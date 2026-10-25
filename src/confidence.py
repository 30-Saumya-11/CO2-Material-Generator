import numpy as np
from utils import load_config

def check_co2_range(target_co2, config=None):
    """
    Checks the target CO2 against the training boundaries.
    Returns: (level, warning_message)
    """
    if config is None:
        config = load_config()
        
    co2_min = config['confidence']['co2_min']
    co2_max = config['confidence']['co2_max']
    warn_buffer = config['confidence']['warn_buffer']
    hypothesis_limit = config['confidence']['hypothesis_limit']
    
    co2_range = co2_max - co2_min
    buffer_val = co2_range * warn_buffer
    hyp_val = co2_range * hypothesis_limit
    
    if co2_min <= target_co2 <= co2_max:
        if target_co2 <= (co2_min + buffer_val) or target_co2 >= (co2_max - buffer_val):
            return "SAFE", "Target is near training boundaries. Predictions have moderate confidence."
        else:
            return "SAFE", "Target is well within training boundaries. Predictions are highly reliable."
    elif (co2_min - hyp_val) <= target_co2 < co2_min or co2_max < target_co2 <= (co2_max + hyp_val):
        return "HYPOTHESIS", f"Target ({target_co2:.2f} mmol/g) is within 20% extrapolation range. Flagged as unvalidated hypothesis. Recommend lab validation."
    else:
        return "SPECULATIVE", f"Target ({target_co2:.2f} mmol/g) is far beyond training range (>20% extrapolation). Predictions are highly speculative and unvalidated."

def evaluate_material_confidence(target_co2, candidate_properties, validation_co2, config=None):
    """
    Evaluates the overall confidence score (0-100%) and feature-level confidence.
    candidate_properties: dictionary of the 9 property values in original units
    """
    if config is None:
        config = load_config()
        
    # 1. Input CO2 Range Confidence
    level, message = check_co2_range(target_co2, config)
    if level == "SAFE":
        if "near" in message:
            input_score = 90.0
        else:
            input_score = 100.0
    elif level == "HYPOTHESIS":
        input_score = 60.0
    else:
        input_score = 30.0
        
    # 2. Feature-level boundary check
    feature_ranges = config['feature_ranges']
    out_of_bounds = []
    in_bounds_count = 0
    total_features = 9  # 9 input properties
    
    for feat, limits in feature_ranges.items():
        if feat == 'CO2':
            continue
        val = candidate_properties.get(feat)
        if val is None:
            continue
        if val < limits['min'] or val > limits['max']:
            out_of_bounds.append({
                "feature": feat,
                "value": float(val),
                "min": float(limits['min']),
                "max": float(limits['max']),
                "unit": limits.get('unit', '')
            })
        else:
            in_bounds_count += 1
            
    feature_score = (in_bounds_count / total_features) * 100.0
    
    # 3. Validation Score (percentage error of predicted CO2 vs target CO2)
    val_error_pct = abs(validation_co2 - target_co2) / max(0.01, target_co2) * 100.0
    # Map error to 0-100: deduct 2% for every 1% error
    val_score = max(0.0, 100.0 - 2.0 * val_error_pct)
    
    # Combine scores using weighted average
    overall_score = 0.4 * input_score + 0.3 * feature_score + 0.3 * val_score
    overall_score = np.clip(overall_score, 0.0, 100.0)
    
    return {
        "level": level,
        "overall_score": float(overall_score),
        "validation_error_pct": float(val_error_pct),
        "input_score": float(input_score),
        "feature_score": float(feature_score),
        "validation_score": float(val_score),
        "out_of_bounds_features": out_of_bounds,
        "message": message
    }
