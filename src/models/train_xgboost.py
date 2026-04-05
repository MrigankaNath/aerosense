
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import joblib
import os

# ── Feature columns the model will use ──────────────────────────
FEATURE_COLS = [
    # Satellite
    "aod_047_mean", "aod_055_mean", "aod_047_max",
    "no2_total_mean", "no2_trop_mean", "no2_total_max",
    "co_mean", "co_max",
    # Meteorological (ERA5)
    "temp_c", "wind_speed", "relative_humidity",
    "blh_mean", "blh_min", "blh_max",

    # Spatial
    "latitude", "longitude",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "is_igp", "is_coastal",
    # Temporal
    "hour", "month", "day_of_week",
    "stubble_season", "is_rush_hour",
    # PM2.5 range (useful proxy)
    "pm25_min", "pm25_max",
]

TARGET_COL = "pm25"


def load_and_prepare(path: str):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")

    # Fill missing values with median imputation
    fill_cols = [
        "aod_047_mean", "aod_055_mean", "aod_047_max",
        "temp_c", "wind_speed", "relative_humidity",
        "blh_mean", "blh_min", "blh_max",
        "sp_mean", "tp_mean",
    ]
    for col in fill_cols:
        if col in df.columns:
            median = df[col].median()
            filled = df[col].isna().sum()
            df[col] = df[col].fillna(median)
            if filled > 0:
                print(f"  Filled {col}: {filled} NaN → median {median:.3f}")

    # Only keep feature cols that exist in the dataframe
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"\n⚠️  Missing feature columns (skipped): {missing_cols}")

    X = df[available]
    y = df[TARGET_COL]

    print(f"\nFeatures used: {len(available)} | Target: {TARGET_COL}")
    print(f"PM2.5 range: {y.min():.1f} – {y.max():.1f} µg/m³")
    return X, y, df


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n{'='*40}")
    print(f"  RMSE : {rmse:.2f} µg/m³")
    print(f"  MAE  : {mae:.2f} µg/m³")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*40}")

    def aqi_cat(v):
        if v <= 30:   return "Good"
        elif v <= 60: return "Satisfactory"
        elif v <= 90: return "Moderate"
        elif v <= 120: return "Poor"
        elif v <= 250: return "Very Poor"
        else:          return "Severe"

    actual_cats = [aqi_cat(v) for v in y_test]
    pred_cats   = [aqi_cat(v) for v in preds]
    cat_acc = sum(a == p for a, p in zip(actual_cats, pred_cats)) / len(actual_cats)
    print(f"  AQI Category Accuracy: {cat_acc*100:.1f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "cat_acc": cat_acc, "preds": preds}


def run_shap(model, X_test):
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("\nTop feature importances (mean |SHAP|):")
    importance = pd.DataFrame({
        "feature":   X_test.columns,
        "mean_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)
    print(importance.head(15).to_string(index=False))

    return shap_values, explainer


if __name__ == "__main__":
    os.makedirs("data/models", exist_ok=True)

    X, y, df = load_and_prepare("data/processed/master_features.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train)

    results = evaluate(model, X_test, y_test)

    shap_vals, explainer = run_shap(model, X_test)

    joblib.dump(model,     "data/models/xgboost_pm25.pkl")
    joblib.dump(explainer, "data/models/shap_explainer.pkl")
    print("\n💾 Model saved to data/models/xgboost_pm25.pkl")
    print("💾 Explainer saved to data/models/shap_explainer.pkl")