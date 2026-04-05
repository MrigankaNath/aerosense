import cdsapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import zipfile
import os

load_dotenv()

CDS_API_KEY = os.getenv("CDS_API_KEY")
CDS_API_URL = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")


def init_cds():
    client = cdsapi.Client(
        url=CDS_API_URL,
        key=CDS_API_KEY,
        quiet=False
    )
    print("✅ CDS client initialized")
    return client


def fetch_era5_india(year: int, month: int, day: int, output_path: str):
    """
    Fetch ERA5 meteorological variables over India for a single day.
    Skips download if file already exists.
    """
    if os.path.exists(output_path):
        print(f"✅ ERA5 file already exists: {output_path} — skipping download")
        return

    client = init_cds()
    print(f"Fetching ERA5 for {year}-{month:02d}-{day:02d}...")

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "total_precipitation",
                "boundary_layer_height",
                "high_vegetation_cover",
                "low_vegetation_cover",
            ],
            "year":  str(year),
            "month": f"{month:02d}",
            "day":   f"{day:02d}",
            "time":  ["00:00", "06:00", "12:00", "18:00"],
            "area":  [37, 68, 8, 97],
            "format": "netcdf.zip",
        },
        output_path
    )
    print(f"✅ ERA5 downloaded to {output_path}")


def unzip_era5(zip_path: str) -> str:
    """
    Unzip ERA5 zip file and return path to the extracted .nc file.
    Skips if already extracted.
    """
    extract_dir = os.path.dirname(zip_path)
    nc_path = zip_path.replace(".zip", ".nc")

    if os.path.exists(nc_path):
        print(f"✅ Already extracted: {nc_path}")
        return nc_path

    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print(f"  Files in zip: {names}")
        z.extractall(extract_dir)
        # Find the .nc file
        for name in names:
            if name.endswith(".nc"):
                extracted = os.path.join(extract_dir, name)
                # Rename to standard name
                os.rename(extracted, nc_path)
                print(f"✅ Extracted to {nc_path}")
                return nc_path

    print("❌ No .nc file found in zip")
    return ""


def extract_era5_at_stations(nc_path: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ERA5 values at CPCB station locations using xarray.
    """
    import xarray as xr

    print(f"\nExtracting ERA5 values at {len(stations_df)} stations...")

    ds = None
    for engine in ["netcdf4", "scipy"]:
        try:
            ds = xr.open_dataset(nc_path, engine=engine)
            print(f"✅ Opened with engine: {engine}")
            break
        except Exception as e:
            print(f"  ⚠️  Engine {engine} failed: {e}")

    if ds is None:
        print("❌ Could not open ERA5 file.")
        return pd.DataFrame()

    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")

    # Detect coordinate names
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    rows = []
    for _, station in stations_df.iterrows():
        row = {
            "station":   station["station"],
            "city":      station.get("city", ""),
            "latitude":  station["latitude"],
            "longitude": station["longitude"],
        }

        for var in ds.data_vars:
            try:
                val = ds[var].sel(
                    {lat_name: station["latitude"],
                     lon_name: station["longitude"]},
                    method="nearest"
                ).values

                if val.ndim > 0:
                    row[f"{var}_mean"] = float(np.nanmean(val))
                    row[f"{var}_min"]  = float(np.nanmin(val))
                    row[f"{var}_max"]  = float(np.nanmax(val))
                else:
                    row[f"{var}_mean"] = float(val)

            except Exception:
                continue

        # Derived: temperature Celsius
        for t_var in ["t2m_mean", "VAR_2T_mean"]:
            if t_var in row:
                row["temp_c"] = row[t_var] - 273.15
                break

        # Derived: wind speed
        for u_var, v_var in [("u10_mean", "v10_mean"), ("VAR_10U_mean", "VAR_10V_mean")]:
            if u_var in row and v_var in row:
                row["wind_speed"] = np.sqrt(row[u_var]**2 + row[v_var]**2)
                break

        # Derived: relative humidity
        for dew_var, t_var in [("d2m_mean", "t2m_mean"), ("VAR_2D_mean", "VAR_2T_mean")]:
            if dew_var in row and t_var in row:
                td = row[dew_var] - 273.15
                t  = row[t_var]   - 273.15
                row["relative_humidity"] = 100 * (
                    np.exp(17.625 * td / (243.04 + td)) /
                    np.exp(17.625 * t  / (243.04 + t))
                )
                break

        rows.append(row)

    ds.close()
    df = pd.DataFrame(rows)
    print(f"✅ Extracted ERA5 for {len(df)} stations")
    return df


if __name__ == "__main__":
    os.makedirs("data/raw/era5", exist_ok=True)

    # ERA5 has ~5 day lag so fetch 5 days ago
    target_date = datetime.now() - timedelta(days=5)
    y = target_date.year
    m = target_date.month
    d = target_date.day

    zip_path = f"data/raw/era5/era5_{y}{m:02d}{d:02d}.zip"

    # Step 1 — download
    fetch_era5_india(y, m, d, zip_path)

    # Step 2 — unzip
    nc_path = unzip_era5(zip_path)
    if not nc_path:
        print("❌ Extraction failed. Exiting.")
        exit(1)

    # Step 3 — extract at station locations
    stations = pd.read_csv("data/processed/cpcb_pm25_clean.csv")
    era5_df  = extract_era5_at_stations(nc_path, stations)

    if not era5_df.empty:
        out_path = "data/raw/era5/era5_stations.csv"
        era5_df.to_csv(out_path, index=False)
        print(f"\n💾 Saved to {out_path}")

        # Show key columns
        show_cols = ["station", "city"]
        for col in ["temp_c", "wind_speed", "relative_humidity", "blh_mean"]:
            if col in era5_df.columns:
                show_cols.append(col)

        print(f"\nSample meteorological data:")
        print(era5_df[show_cols].head(10).to_string())