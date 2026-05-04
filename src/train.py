import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("data/training_data.csv")

X = df[["temperature_c", "building_sqm", "occupancy_pct", "is_weekday"]]
y = df["energy_kwh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor()

mlflow.set_experiment("powergrid-energy-kwh")

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # IMPORTANT
    mlflow.sklearn.log_model(model, "model")

    print("Training done. MAE:", mae)
