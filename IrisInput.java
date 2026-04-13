from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once at startup
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    features: list[float]  # Expects exactly 4 features

@app.get("/")
def health_check():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: IrisInput):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": int(pred[0])}