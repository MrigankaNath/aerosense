from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.predictor import predict

app = FastAPI(
    title="AeroSense India API",
    description="Hyperlocal PM2.5 prediction for India — explainable, uncertainty-aware",
    version="1.0.0"
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    latitude:          float
    longitude:         float
    # Optional — if not provided, training medians are used
    aod_047_mean:      Optional[float] = None
    no2_total_mean:    Optional[float] = None
    co_mean:           Optional[float] = None
    temp_c:            Optional[float] = None
    wind_speed:        Optional[float] = None
    relative_humidity: Optional[float] = None
    blh_mean:          Optional[float] = None
    hour:              Optional[int]   = None
    month:             Optional[int]   = None


@app.get("/")
def root():
    return {
        "name":    "AeroSense India API",
        "version": "1.0.0",
        "status":  "operational",
        "endpoints": ["/predict", "/stations", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/predict")
def predict_pm25(request: PredictionRequest):
    """
    Predict PM2.5 at any location in India.
    Returns prediction, AQI category, health advisory and SHAP explanation.
    """
    try:
        now = datetime.now()
        data = request.dict()

        # Auto-fill time features if not provided
        if data["hour"] is None:
            data["hour"] = now.hour
        if data["month"] is None:
            data["month"] = now.month

        data["day_of_week"] = now.weekday()

        result = predict(data)
        result["location"] = {
            "latitude":  request.latitude,
            "longitude": request.longitude,
        }
        result["timestamp"] = now.isoformat()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stations")
def get_stations():
    """
    Return all CPCB stations with latest PM2.5 readings.
    Used for the map layer on the frontend.
    """
    try:
        df = pd.read_csv("data/processed/cpcb_pm25_clean.csv")
        df = df[["station", "city", "state",
                 "latitude", "longitude",
                 "pm25", "aqi_category", "timestamp"]]
        df = df.dropna(subset=["latitude", "longitude", "pm25"])
        return {
            "count":    len(df),
            "stations": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stations/{station_name}/explain")
def explain_station(station_name: str):
    """
    Get SHAP explanation for a specific CPCB station.
    """
    try:
        df = pd.read_csv("data/processed/master_features.csv")
        station = df[df["station"].str.contains(station_name, case=False)]

        if station.empty:
            raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")

        row = station.iloc[0].to_dict()
        result = predict(row)
        result["station"] = row.get("station")
        result["city"]    = row.get("city")
        result["state"]   = row.get("state")
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))