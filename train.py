import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_and_log_model():
    # Set MLflow experiment
    mlflow.set_experiment("Iris_Classification_Experiment")
    
    with mlflow.start_run():
        # Load data
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define hyperparameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="IrisClassifier"
        )
        joblib.dump(model, "model.joblib")
        
        print(f"Model trained & logged to MLflow!")
        print(f"Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_and_log_model()