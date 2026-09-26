import mlflow
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000")


MODEL_URI = "models:/CustomerChurnModel@champion"


model = mlflow.pyfunc.load_model(MODEL_URI)


sample = pd.DataFrame([
    {
        "age": 35,
        "monthly_income": 5000,
        "tenure": 12,
        "support_calls": 2,
        "contract_months": 12    
    }
])


prediction = model.predict(sample)

if prediction[0] == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is likely to stay.")