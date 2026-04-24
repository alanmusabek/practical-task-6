# SIS 3
## What's changed?
    Integrated MLFlow.
    Added Frontend UI to ease work with the ML model for better user-experience.
## Description
    This project focuses on imulating the production process: workflow and lifecycle of ML model. MLFlow is used for managing ML models, experimenting, and other related stuff.
## Instructions
### 0. Install Requirements
```
pip install -r requirements.txt
```
### 1. Train model
```
python train.py
```
#### 2. Run API
```
uvicorn main:app --reload
```
#### 3. Run FrontendUI
```
streamlit run streamlit_app.py
```
#### 4. Run MLFlow
```
mlflow ui
```
#### 5. Docker build
```
docker build -t ml-fastapi-app .
```
#### 6. Run container
```
docker run -p 8000:8000 -v ${PWD}/mlruns:/app/mlruns ml-backend 
```


---------------------
# (LEGACY) PRACTICAL TASK 6
## DESCRIPTION
This project is a simulation of a real world scenario of ML model deployment in production. We start from creating and training a simple ML model, then we create API logic for the purpose of allowing a user/client to interact with our model. After it, we package the model, using Docker to containeraize our application. In betweens we check and test correctness of our work.
## STRUCTURE AND INSTRUCTIONS
The project follows the next structure:
    <p>train.py <- train and save the model</p>
    <p>main.py <- the main workspace, it is started by using fastAPI or docker</p>
    <p>model.joblib <- saved model</p>
    <p>Dockerfile <- docker file</p>
### STEPS
#### 1. Train model
```
python train.py
```
#### 2. Run API
```
uvicorn main:app --reload
```
#### 3. Docker build
```
docker build -t ml-fastapi-app .
```
#### 4. Run container
```
docker run -p 8000:8000 ml-fastapi-app
```