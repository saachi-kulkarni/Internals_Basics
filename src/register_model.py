import mlflow
import json

MODEL_NAME = "powergrid-energy-kwh-predictor-"

# Get latest run
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("powergrid-energy-kwh")
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])

run_id = runs[0].info.run_id

model_uri = f"runs:/{run_id}/model"

# Register model
result = mlflow.register_model(model_uri, MODEL_NAME)

version = result.version

# Get metric
mae = runs[0].data.metrics.get("mae", 0)

output = {
    "registered_model_name": MODEL_NAME,
    "version": version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": mae
}

with open("results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Registered Model Version:", version)
print("Saved to results/step4_s6.json")

