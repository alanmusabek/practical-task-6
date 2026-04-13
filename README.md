# PRACTICAL TASK 6
## DESCRIPTION
This project is a simulation of a real world scenario of ML model deployment in production 
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