from mlflow import MlflowClient

c = MlflowClient('http://127.0.0.1:5000')
mv = c.get_model_version_by_alias('CustomerChurnModel', 'champion')
print("Version:", mv.version)
print("Source:", mv.source)
print("Model ID:", mv.model_id)
print("Run ID:", mv.run_id)