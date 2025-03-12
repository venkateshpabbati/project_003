import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlruns.db")
model = mlflow.sklearn.load_model("mlruns/0/model")

app = FastAPI()

class Features(BaseModel):
    data: list

@app.post("/predict")
def predict(features: Features):
    input_data = np.array(features.data).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}