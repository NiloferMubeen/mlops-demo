import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

exp_id = mlflow.create_experiment("customer-churn-v3")
exp = mlflow.get_experiment(exp_id)
print("Experiment ID:", exp.experiment_id)
print("Artifact location:", exp.artifact_location)