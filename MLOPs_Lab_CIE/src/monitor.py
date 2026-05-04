import json
import pandas as pd

# Load training data
train_df = pd.read_csv("data/training_data.csv")

# Load logs
with open("logs/predictions.jsonl", "r") as f:
    logs = [json.loads(line) for line in f]

# Convert logs to DataFrame
log_df = pd.DataFrame([l["input"] for l in logs])

# Total predictions
total_predictions = len(logs)

# Mean prediction
mean_prediction = sum(l["prediction"] for l in logs) / total_predictions

alerts = []

# -------- Temperature Drift --------
train_mean_temp = train_df["temperature_c"].mean()
live_mean_temp = log_df["temperature_c"].mean()

shift_temp = abs(train_mean_temp - live_mean_temp)

alerts.append({
    "feature": "temperature_c",
    "train_mean": round(train_mean_temp, 2),
    "live_mean": round(live_mean_temp, 2),
    "shift": round(shift_temp, 2),
    "threshold": 4.71,
    "status": "ALERT" if shift_temp > 4.71 else "OK"
})

# -------- Occupancy Drift --------
train_mean_occ = train_df["occupancy_pct"].mean()
live_mean_occ = log_df["occupancy_pct"].mean()

shift_occ = abs(train_mean_occ - live_mean_occ)

alerts.append({
    "feature": "occupancy_pct",
    "train_mean": round(train_mean_occ, 2),
    "live_mean": round(live_mean_occ, 2),
    "shift": round(shift_occ, 2),
    "threshold": 17.9,
    "status": "ALERT" if shift_occ > 17.9 else "OK"
})

# Drift detected?
drift_detected = any(a["status"] == "ALERT" for a in alerts)

# Final output
output = {
    "total_predictions": total_predictions,
    "mean_prediction": round(mean_prediction, 2),
    "drift_detected": drift_detected,
    "alerts": alerts
}

# Save result
with open("results/step3_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Monitoring complete!")
print(json.dumps(output, indent=4))
