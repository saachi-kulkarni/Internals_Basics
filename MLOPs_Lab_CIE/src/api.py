from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
from datetime import datetime

app = FastAPI()

model = joblib.load("models/model.pkl")

LOG_FILE = "logs/predictions.jsonl"
os.makedirs("logs", exist_ok=True)

class Input(BaseModel):
    temperature_c: float
    building_sqm: float
    occupancy_pct: float
    is_weekday: int

@app.get("/status")
def status():
    return {"status": "operational", "service": "PowerGrid API"}

@app.post("/estimate")
def estimate(data: Input):
    arr = [[
        data.temperature_c,
        data.building_sqm,
        data.occupancy_pct,
        data.is_weekday
    ]]

    pred = float(model.predict(arr)[0])

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": data.dict(),
        "prediction": pred
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": pred}
