from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/model.pkl")

# Create app
app = FastAPI()

# Input schema
class Input(BaseModel):
    temperature_c: float
    building_sqm: float
    occupancy_pct: float
    is_weekday: int

# Health check
@app.get("/status")
def status():
    return {
        "status": "operational",
        "service": "PowerGrid API"
    }

# Prediction endpoint
@app.post("/estimate")
def estimate(data: Input):
    arr = np.array([[data.temperature_c,
                     data.building_sqm,
                     data.occupancy_pct,
                     data.is_weekday]])

    pred = model.predict(arr)[0]

    return {
        "prediction": float(pred)
    }
