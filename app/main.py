from fastapi import FastAPI
import mlflow
import pandas as pd


# Connect to MLflow
mlflow.set_tracking_uri("host.docker.internal:5000")


# Load the champion model
MODEL_URI = "models:/CustomerChurnModel@champion"

model = mlflow.pyfunc.load_model(MODEL_URI)


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CustomerChurnModel@champion"
    }


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([{
        "age": data["age"],
        "monthly_income": data["monthly_income"],
        "tenure": data["tenure"],
        "support_calls": data["support_calls"],
        "contract_months": data["contract_months"]
    }])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }
