import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

df = pd.read_csv("data/training_data.csv")

X = df.drop("energy_kwh", axis=1)
y = df["energy_kwh"]

models = {
    "Ridge": Ridge(),
    "GradientBoosting": GradientBoostingRegressor()
}

results = []

mlflow.set_experiment("powergrid-energy-kwh")

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X, y)
        preds = model.predict(X)

        mae = mean_absolute_error(y, preds)
        rmse = mean_squared_error(y, preds) ** 0.5
        r2 = r2_score(y, preds)
        mape = (abs((y - preds) / y)).mean()

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

best = min(results, key=lambda x: x["mae"])

best_model = models[best["name"]]
best_model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump({
        "experiment_name": "powergrid-energy-kwh",
        "models": results,
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"]
    }, f, indent=4)

print("✅ Task 1 complete")
