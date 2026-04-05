
import pandas as pd
import numpy as np
import joblib
import shap
import os

# Load model and explainer once at startup
MODEL_PATH     = "data/models/xgboost_pm25.pkl"
EXPLAINER_PATH = "data/models/shap_explainer.pkl"

model     = joblib.load(MODEL_PATH)
explainer = joblib.load(EXPLAINER_PATH)

FEATURE_COLS = [
    "aod_047_mean", "aod_055_mean", "aod_047_max",
    "no2_total_mean", "no2_trop_mean", "no2_total_max",
    "co_mean", "co_max",
    "temp_c", "wind_speed", "relative_humidity",
    "blh_mean", "blh_min", "blh_max",
    "latitude", "longitude",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "is_igp", "is_coastal",
    "hour", "month", "day_of_week",
    "stubble_season", "is_rush_hour",
    "pm25_min", "pm25_max",
]

# Medians for imputation (from training data)
_df = pd.read_csv("data/processed/master_features.csv")
MEDIANS = {col: _df[col].median() for col in FEATURE_COLS if col in _df.columns}


def aqi_category(pm25: float) -> str:
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"


def health_advisory(pm25: float) -> str:
    if pm25 <= 30:
        return "Air quality is good. Enjoy outdoor activities."
    elif pm25 <= 60:
        return "Air quality is satisfactory. Sensitive individuals should limit prolonged outdoor exertion."
    elif pm25 <= 90:
        return "Moderate pollution. People with respiratory issues should reduce outdoor activity."
    elif pm25 <= 120:
        return "Poor air quality. Avoid prolonged outdoor activity. Use N95 mask if going out."
    elif pm25 <= 250:
        return "Very poor air quality. Stay indoors. Keep windows closed. N95 mask essential outdoors."
    else:
        return "Severe pollution — health emergency level. Avoid all outdoor activity. Seek medical attention if experiencing symptoms."


def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Build a feature vector from input data.
    Missing features are filled with training medians.
    """
    row = {}
    for col in FEATURE_COLS:
        val = data.get(col)
        if val is None:
            val = MEDIANS.get(col, 0.0)
        row[col] = val

    # Derive spatial encoding if raw lat/lon provided
    if "latitude" in data and "longitude" in data:
        lat = data["latitude"]
        lon = data["longitude"]
        row["lat_sin"] = np.sin(np.radians(lat))
        row["lat_cos"] = np.cos(np.radians(lat))
        row["lon_sin"] = np.sin(np.radians(lon))
        row["lon_cos"] = np.cos(np.radians(lon))
        row["is_igp"]  = int(24 <= lat <= 32 and 73 <= lon <= 88)
        row["is_coastal"] = int(lat < 15 or lon < 73 or lon > 90)

    return pd.DataFrame([row])[FEATURE_COLS]


def predict(data: dict) -> dict:
    """
    Run PM2.5 prediction and return full result with explanation.
    """
    X = build_feature_vector(data)

    # Prediction
    pm25_pred = float(np.clip(model.predict(X)[0], 0, None))

    # SHAP explanation
    shap_values = explainer.shap_values(X)[0]
    base_value  = float(explainer.expected_value)

    # Top 5 contributing features
    shap_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap":    shap_values,
        "value":   X.iloc[0].values
    }).reindex(pd.Series(shap_values).abs().sort_values(ascending=False).index)

    top_factors = []
    for _, row in shap_df.head(5).iterrows():
        direction = "increases" if row["shap"] > 0 else "decreases"
        top_factors.append({
            "feature":   row["feature"],
            "value":     round(float(row["value"]), 4),
            "shap":      round(float(row["shap"]), 3),
            "direction": direction
        })

    # Generate plain-English explanation
    explanation = generate_explanation(pm25_pred, top_factors, data)

    return {
        "pm25":           round(pm25_pred, 1),
        "aqi_category":   aqi_category(pm25_pred),
        "health_advisory": health_advisory(pm25_pred),
        "base_pm25":      round(base_value, 1),
        "top_factors":    top_factors,
        "explanation":    explanation,
    }


def generate_explanation(pm25: float, factors: list, data: dict) -> str:
    """
    Generate a plain-English explanation of why PM2.5 is at this level.
    Rule-based — no LLM needed.
    """
    lines = [f"Predicted PM2.5: {pm25:.1f} µg/m³ ({aqi_category(pm25)})"]
    lines.append("\nKey drivers:")

    factor_map = {
        "blh_mean":         lambda v: f"Low boundary layer height ({v:.0f}m) is trapping pollutants close to ground" if v < 500 else f"High boundary layer ({v:.0f}m) is helping disperse pollution",
        "no2_total_mean":   lambda v: f"Elevated NO₂ ({v*1e6:.2f}×10⁻⁶ mol/cm²) indicating heavy traffic or industrial activity",
        "co_mean":          lambda v: f"Elevated CO ({v:.3f} mol/m²) suggesting biomass burning or vehicle emissions",
        "aod_047_mean":     lambda v: f"High aerosol optical depth ({v:.3f}) indicating dense particulate matter in atmosphere",
        "wind_speed":       lambda v: f"Low wind speed ({v:.1f} m/s) reducing pollutant dispersal" if v < 2 else f"Wind speed ({v:.1f} m/s) helping disperse pollutants",
        "relative_humidity": lambda v: f"High humidity ({v:.0f}%) causing hygroscopic PM2.5 growth" if v > 70 else f"Low humidity ({v:.0f}%) — no rain washout",
        "temp_c":           lambda v: f"Temperature {v:.1f}°C affecting atmospheric stability",
        "is_igp":           lambda v: "Location in Indo-Gangetic Plain — historically highest pollution zone in India" if v == 1 else None,
        "stubble_season":   lambda v: "Stubble burning season in Punjab/Haryana — agricultural fires contributing significantly" if v == 1 else None,
        "is_rush_hour":     lambda v: "Rush hour — peak vehicular emissions" if v == 1 else None,
        "latitude":         lambda v: f"Geographic position (lat {v:.2f}°)",
        "pm25_max":         lambda v: f"Recent peak PM2.5 of {v:.0f} µg/m³ at this station",
    }

    for factor in factors:
        feat = factor["feature"]
        val  = factor["value"]
        if feat in factor_map:
            try:
                desc = factor_map[feat](val)
                if desc:
                    arrow = "↑" if factor["shap"] > 0 else "↓"
                    lines.append(f"  {arrow} {desc}")
            except Exception:
                continue

    return "\n".join(lines)