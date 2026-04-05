
import pandas as pd
import numpy as np
from datetime import datetime

def load_all_sources() -> dict:
    """Load all data sources."""
    print("Loading data sources...")
    
    cpcb = pd.read_csv("data/processed/cpcb_pm25_clean.csv")
    aod  = pd.read_csv("data/raw/modis_aod.csv")
    no2  = pd.read_csv("data/raw/sentinel5p_no2.csv")
    co   = pd.read_csv("data/raw/sentinel5p_co.csv")

    print(f"  CPCB stations:  {len(cpcb)}")
    print(f"  MODIS AOD:      {len(aod)}")
    print(f"  Sentinel NO2:   {len(no2)}")
    print(f"  Sentinel CO:    {len(co)}")

    return {"cpcb": cpcb, "aod": aod, "no2": no2, "co": co}


def merge_satellite_to_cpcb(data: dict) -> pd.DataFrame:
    """
    Merge satellite data onto CPCB stations by station name.
    Since CPCB is a snapshot (no date), we use mean satellite
    values per station as static features for now.
    Later when we have historical CPCB data, we'll merge on date too.
    """
    print("\nMerging satellite features onto CPCB stations...")

    cpcb = data["cpcb"].copy()

    # Aggregate satellite data per station (mean across dates)
    aod_agg = data["aod"].groupby("station").agg(
        aod_047_mean=("aod_047", "mean"),
        aod_055_mean=("aod_055", "mean"),
        aod_047_max=("aod_047", "max"),
    ).reset_index()

    no2_agg = data["no2"].groupby("station").agg(
        no2_total_mean=("no2_total", "mean"),
        no2_trop_mean=("no2_trop",  "mean"),
        no2_total_max=("no2_total", "max"),
    ).reset_index()

    co_agg = data["co"].groupby("station").agg(
        co_mean=("co", "mean"),
        co_max=("co", "max"),
    ).reset_index()

    # Merge onto CPCB
    df = cpcb.merge(aod_agg, on="station", how="left")
    df = df.merge(no2_agg, on="station", how="left")
    df = df.merge(co_agg,  on="station", how="left")

    print(f"  Stations with AOD data:  {df['aod_047_mean'].notna().sum()}")
    print(f"  Stations with NO2 data:  {df['no2_total_mean'].notna().sum()}")
    print(f"  Stations with CO data:   {df['co_mean'].notna().sum()}")

    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial features:
    - Lat/lon encoded as sin/cos (better than raw for ML)
    - IGP belt flag (Indo-Gangetic Plain — highest pollution zone)
    - Coastal flag
    """
    print("\nAdding spatial features...")

    # Sin/cos encoding of lat/lon
    df["lat_sin"] = np.sin(np.radians(df["latitude"]))
    df["lat_cos"] = np.cos(np.radians(df["latitude"]))
    df["lon_sin"] = np.sin(np.radians(df["longitude"]))
    df["lon_cos"] = np.cos(np.radians(df["longitude"]))

    # IGP belt: roughly lat 24-32, lon 73-88
    df["is_igp"] = (
        (df["latitude"].between(24, 32)) &
        (df["longitude"].between(73, 88))
    ).astype(int)

    # Coastal: within ~100km of coast (simplified)
    df["is_coastal"] = (
        (df["latitude"] < 15) |
        (df["longitude"] < 73) |
        (df["longitude"] > 90)
    ).astype(int)

    print(f"  IGP belt stations:  {df['is_igp'].sum()}")
    print(f"  Coastal stations:   {df['is_coastal'].sum()}")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features from timestamp.
    """
    print("\nAdding temporal features...")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]      = df["timestamp"].dt.hour
    df["month"]     = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Season (India-specific)
    def get_season(month):
        if month in [12, 1, 2]:  return "winter"
        elif month in [3, 4, 5]: return "summer"
        elif month in [6, 7, 8, 9]: return "monsoon"
        else: return "post_monsoon"

    df["season"] = df["month"].apply(get_season)

    # Stubble burning season (Oct-Nov in Punjab/Haryana)
    df["stubble_season"] = (
        (df["month"].isin([10, 11])) &
        (df["state"].isin(["Punjab", "Haryana"]))
    ).astype(int)

    # Morning rush hour
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    print(f"  Season distribution:\n{df['season'].value_counts().to_string()}")

    return df


def add_aqi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PM2.5 derived features useful for model training."""
    print("\nAdding AQI features...")

    df["pm25_log"] = np.log1p(df["pm25"])   # log transform reduces skew
    df["pm25_zscore"] = (df["pm25"] - df["pm25"].mean()) / df["pm25"].std()
    df["is_severe"] = (df["pm25"] > 150).astype(int)

    return df


def build_master_features() -> pd.DataFrame:
    """Full pipeline — loads, merges, engineers all features."""
    data = load_all_sources()
    df   = merge_satellite_to_cpcb(data)
    df   = add_spatial_features(df)
    df   = add_temporal_features(df)
    df   = add_aqi_features(df)

    print(f"\n✅ Master feature table: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    return df


if __name__ == "__main__":
    df = build_master_features()
    df.to_csv("data/processed/master_features.csv", index=False)
    print(f"\n💾 Saved to data/processed/master_features.csv")
    print(f"\nSample:")
    print(df.head(5).to_string())