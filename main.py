from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow
import os

app = FastAPI()

# Try to load model from MLflow Registry, fallback to local file
def load_model():
    try:
        # Try loading from MLflow Model Registry
        model_uri = "models:/IrisClassifier/Production"
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded from MLflow Registry")
        return model
    except:
        # Fallback to local file
        print("Loading local model.joblib")
        return joblib.load("model.joblib")

model = load_model()

class IrisInput(BaseModel):
    features: list[float]  # [sepal_length, sepal_width, petal_length, petal_width]

@app.get("/")
def health_check():
    return {"message": "ML API is running", "model_source": "MLflow or local"}

@app.post("/predict")
def predict(data: IrisInput):
    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Please provide exactly 4 features")
    
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    proba = model.predict_proba(input_array)[0].tolist()
    
    return {
        "prediction": int(prediction[0]),
        "class_probabilities": {
            "setosa": round(proba[0], 4),
            "versicolor": round(proba[1], 4),
            "virginica": round(proba[2], 4)
        }
    }

@app.get("/model-info")
def model_info():
    """Return metadata about the loaded model"""
    return {
        "model_name": "IrisClassifier",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa (0)", "versicolor (1)", "virginica (2)"]
    }