import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


DATASET_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_KEY     = os.getenv("CPCB_API_KEY")
RAW_PATH    = "data/raw/cpcb_raw.csv"


def download_csv() -> pd.DataFrame:
    """
    Downloads the latest CPCB snapshot from data.gov.in.
    No API key required as of now. Using this for training data.
    """
    print(f"[{datetime.now()}] Downloading CPCB snapshot...")

    url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?format=csv&limit=5000"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(RAW_PATH, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(RAW_PATH)
        print(f"Downloaded {len(df)} records → {RAW_PATH}")
        return df

    except Exception as e:
        print(f"⚠️  Auto-download failed ({e})")
        print("👉 Manual fallback: download CSV from data.gov.in → save as data/raw/cpcb_raw.csv")
        return pd.DataFrame()


def fetch_live_api(limit: int = 500) -> pd.DataFrame:
    """
    Fetches real-time CPCB data via data.gov.in API.
    Requires CPCB_API_KEY in .env
    Use this for live dashboard refresh.
    """
    if not API_KEY:
        print("No CPCB_API_KEY in .env — falling back to download_csv()")
        return download_csv()

    print(f"[{datetime.now()}] Fetching live CPCB data via API...")

    params = {
        "api-key": API_KEY,
        "format":  "json",
        "limit":   limit,
        "fields":  "country,state,city,station,last_update,latitude,longitude,pollutant_id,pollutant_min,pollutant_max,pollutant_avg"
    }

    try:
        response = requests.get(DATASET_URL, params=params, timeout=60)
        response.raise_for_status()

        records = response.json().get("records", [])
        if not records:
            print("No records returned. Falling back to download_csv()")
            return download_csv()

        df = pd.DataFrame(records)
        df.to_csv(RAW_PATH, index=False)
        print(f"Fetched {len(df)} live records → {RAW_PATH}")
        return df

    except requests.exceptions.Timeout:
        print("API timed out. Falling back to download_csv()")
        return download_csv()

    except Exception as e:
        print(f"API error: {e}. Falling back to download_csv()")
        return download_csv()


if __name__ == "__main__":
    # Tries live API first, falls back to CSV download automatically
    df = fetch_live_api()

    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample:")
        print(df.head(5).to_string())