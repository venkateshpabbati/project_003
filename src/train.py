import mlflow
import mlflow.sklearn
# import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_data

# Set up MLflow to track experiments locally
mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Local MLflow tracking
mlflow.set_experiment("linear_regression")

X_train, X_test, y_train, y_test = load_data()

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("MSE", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model trained with MSE: {mse}")