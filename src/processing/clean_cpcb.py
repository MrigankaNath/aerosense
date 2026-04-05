import pandas as pd
from datetime import datetime

def clean_cpcb_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Cleans raw CPCB CSV data.
    - Filters to PM2.5 only
    - Removes invalid readings
    - Standardises column names
    - Saves cleaned version
    """
    df = pd.read_csv(input_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Pollutants available: {df['pollutant_id'].unique()}")

    # Filter PM2.5 only
    pm25 = df[df["pollutant_id"] == "PM2.5"].copy()
    print(f"\nPM2.5 stations: {len(pm25)}")

    # Rename to standard schema
    pm25 = pm25.rename(columns={
        "pollutant_avg": "pm25",
        "pollutant_min": "pm25_min",
        "pollutant_max": "pm25_max",
        "last_update":   "timestamp"
    })

    # Clean types
    pm25["pm25"]      = pd.to_numeric(pm25["pm25"], errors="coerce")
    pm25["pm25_min"]  = pd.to_numeric(pm25["pm25_min"], errors="coerce")
    pm25["pm25_max"]  = pd.to_numeric(pm25["pm25_max"], errors="coerce")
    pm25["latitude"]  = pd.to_numeric(pm25["latitude"], errors="coerce")
    pm25["longitude"] = pd.to_numeric(pm25["longitude"], errors="coerce")
    pm25["timestamp"] = pd.to_datetime(pm25["timestamp"], errors="coerce")

    # Remove invalid rows
    before = len(pm25)
    pm25 = pm25.dropna(subset=["pm25", "latitude", "longitude", "timestamp"])
    pm25 = pm25[pm25["pm25"] > 0]
    pm25 = pm25[pm25["pm25"] < 1000]   # remove sensor errors
    print(f"Removed {before - len(pm25)} invalid rows")

    # Add AQI category (India NAAQS standard)
    def aqi_category(val):
        if val <= 30:   return "Good"
        elif val <= 60: return "Satisfactory"
        elif val <= 90: return "Moderate"
        elif val <= 120: return "Poor"
        elif val <= 250: return "Very Poor"
        else:           return "Severe"

    pm25["aqi_category"] = pm25["pm25"].apply(aqi_category)
    pm25["fetched_at"]   = datetime.now()

    # Final column order
    cols = ["country", "state", "city", "station",
            "latitude", "longitude",
            "pm25", "pm25_min", "pm25_max",
            "aqi_category", "timestamp", "fetched_at"]
    pm25 = pm25[cols]

    pm25.to_csv(output_path, index=False)
    print(f"\n Cleaned data shape: {pm25.shape}")
    print(f"Saved to {output_path}")
    print(f"\nAQI breakdown:")
    print(pm25["aqi_category"].value_counts().to_string())
    print(f"\nMost polluted stations:")
    print(pm25.nlargest(5, "pm25")[["station", "city", "state", "pm25", "aqi_category"]].to_string())
    print(f"\nCleanest stations:")
    print(pm25.nsmallest(5, "pm25")[["station", "city", "state", "pm25", "aqi_category"]].to_string())

    return pm25


if __name__ == "__main__":
    df = clean_cpcb_data(
        input_path="data/raw/cpcb_raw.csv",
        output_path="data/processed/cpcb_pm25_clean.csv"
    )
    print(f"\nSample:")
    print(df.head(10).to_string())