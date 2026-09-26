from mlflow import MlflowClient

c = MlflowClient('http://127.0.0.1:5000')
runs = c.search_runs(experiment_ids=['3'], order_by=['start_time DESC'], max_results=1)
run = runs[0]
print("Run ID:", run.info.run_id)
print("Model outputs:", run.outputs.model_outputs if run.outputs else None)