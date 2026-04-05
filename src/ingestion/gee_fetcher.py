import ee
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()


def init_gee():
    ee.Initialize(project=os.getenv("GEE_PROJECT_ID"))
    print("✅ GEE initialized")


def fetch_modis_aod(start_date: str, end_date: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch MODIS MAIAC AOD (1km) values at CPCB station locations.
    AOD at 470nm is the best satellite proxy for PM2.5.
    """
    print(f"Fetching MODIS MAIAC AOD: {start_date} to {end_date}...")

    india_bbox = ee.Geometry.Rectangle([68, 8, 97, 37])

    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterDate(start_date, end_date)
        .filterBounds(india_bbox)
        .select(["Optical_Depth_047", "Optical_Depth_055"])
    )

    points = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {"station": row["station"], "city": row.get("city", "")}
        )
        for _, row in stations_df.iterrows()
    ])

    def sample_image(image):
        date = image.date().format("YYYY-MM-dd")
        sampled = image.sampleRegions(
            collection=points,
            scale=1000,
            geometries=True
        )
        return sampled.map(lambda f: f.set("date", date))

    sampled_collection = collection.map(sample_image).flatten()

    results = []
    try:
        features = sampled_collection.limit(5000).getInfo()["features"]
        for f in features:
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            results.append({
                "station":  props.get("station"),
                "city":     props.get("city"),
                "longitude": coords[0],
                "latitude":  coords[1],
                "date":     props.get("date"),
                "aod_047":  props.get("Optical_Depth_047"),
                "aod_055":  props.get("Optical_Depth_055"),
            })
    except Exception as e:
        print(f"⚠️  Sampling error: {e}")
        return pd.DataFrame()

    if not results:
        print("⚠️  No AOD data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["aod_047"] = pd.to_numeric(df["aod_047"], errors="coerce") / 1000
    df["aod_055"] = pd.to_numeric(df["aod_055"], errors="coerce") / 1000
    df = df.dropna(subset=["aod_047"])
    df = df[df["aod_047"] > 0]

    print(f"✅ MODIS AOD: {len(df)} station-day records")
    return df


def fetch_sentinel5p_no2(start_date: str, end_date: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch Sentinel-5P NO2 column density at CPCB station locations.
    NO2 is a strong traffic and industrial pollution indicator.
    """
    print(f"Fetching Sentinel-5P NO2: {start_date} to {end_date}...")

    india_bbox = ee.Geometry.Rectangle([68, 8, 97, 37])

    collection = (
        ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
        .filterDate(start_date, end_date)
        .filterBounds(india_bbox)
        .select(["NO2_column_number_density", "tropospheric_NO2_column_number_density"])
    )

    points = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {"station": row["station"], "city": row.get("city", "")}
        )
        for _, row in stations_df.iterrows()
    ])

    def sample_image(image):
        date = image.date().format("YYYY-MM-dd")
        sampled = image.sampleRegions(
            collection=points,
            scale=7000,
            geometries=True
        )
        return sampled.map(lambda f: f.set("date", date))

    sampled_collection = collection.map(sample_image).flatten()

    results = []
    try:
        features = sampled_collection.limit(5000).getInfo()["features"]
        for f in features:
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            results.append({
                "station":   props.get("station"),
                "city":      props.get("city"),
                "longitude": coords[0],
                "latitude":  coords[1],
                "date":      props.get("date"),
                "no2_total": props.get("NO2_column_number_density"),
                "no2_trop":  props.get("tropospheric_NO2_column_number_density"),
            })
    except Exception as e:
        print(f"⚠️  Sampling error: {e}")
        return pd.DataFrame()

    if not results:
        print("⚠️  No NO2 data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["no2_total"] = pd.to_numeric(df["no2_total"], errors="coerce")
    df["no2_trop"]  = pd.to_numeric(df["no2_trop"],  errors="coerce")
    df = df.dropna(subset=["no2_total"])

    print(f"✅ Sentinel-5P NO2: {len(df)} station-day records")
    return df


def fetch_sentinel5p_co(start_date: str, end_date: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch Sentinel-5P CO (Carbon Monoxide) — biomass burning indicator.
    Spikes during stubble burning season in Punjab/Haryana.
    """
    print(f"Fetching Sentinel-5P CO: {start_date} to {end_date}...")

    india_bbox = ee.Geometry.Rectangle([68, 8, 97, 37])

    collection = (
        ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
        .filterDate(start_date, end_date)
        .filterBounds(india_bbox)
        .select(["CO_column_number_density"])
    )

    points = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {"station": row["station"], "city": row.get("city", "")}
        )
        for _, row in stations_df.iterrows()
    ])

    def sample_image(image):
        date = image.date().format("YYYY-MM-dd")
        sampled = image.sampleRegions(
            collection=points,
            scale=7000,
            geometries=True
        )
        return sampled.map(lambda f: f.set("date", date))

    sampled_collection = collection.map(sample_image).flatten()

    results = []
    try:
        features = sampled_collection.limit(5000).getInfo()["features"]
        for f in features:
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            results.append({
                "station":   props.get("station"),
                "city":      props.get("city"),
                "longitude": coords[0],
                "latitude":  coords[1],
                "date":      props.get("date"),
                "co":        props.get("CO_column_number_density"),
            })
    except Exception as e:
        print(f"⚠️  Sampling error: {e}")
        return pd.DataFrame()

    if not results:
        print("⚠️  No CO data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["co"] = pd.to_numeric(df["co"], errors="coerce")
    df = df.dropna(subset=["co"])

    print(f"✅ Sentinel-5P CO: {len(df)} station-day records")
    return df


if __name__ == "__main__":
    # GEE must be initialized first before any ee.* calls
    init_gee()

    # Load cleaned CPCB stations
    stations = pd.read_csv("data/processed/cpcb_pm25_clean.csv")
    print(f"Loaded {len(stations)} CPCB stations\n")

    # Fetch last 7 days
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"Date range: {start} to {end}\n")

    # Fetch AOD
    aod_df = fetch_modis_aod(start, end, stations)
    if not aod_df.empty:
        aod_df.to_csv("data/raw/modis_aod.csv", index=False)
        print(f"💾 AOD saved → data/raw/modis_aod.csv")
        print(aod_df.head())

    print()

    # Fetch NO2
    no2_df = fetch_sentinel5p_no2(start, end, stations)
    if not no2_df.empty:
        no2_df.to_csv("data/raw/sentinel5p_no2.csv", index=False)
        print(f"💾 NO2 saved → data/raw/sentinel5p_no2.csv")
        print(no2_df.head())

    print()

    # Fetch CO
    co_df = fetch_sentinel5p_co(start, end, stations)
    if not co_df.empty:
        co_df.to_csv("data/raw/sentinel5p_co.csv", index=False)
        print(f"💾 CO saved → data/raw/sentinel5p_co.csv")
        print(co_df.head())