import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

exp = mlflow.get_experiment_by_name("customer-churn-v2")
print("Experiment ID:", exp.experiment_id)
print("Artifact location:", exp.artifact_location)