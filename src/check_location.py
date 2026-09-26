from mlflow import MlflowClient
c = MlflowClient('http://127.0.0.1:5000')
lm = c.get_logged_model('m-a9a7c2412f5b401481f178f93567a5bd')
print("Artifact location:", lm.artifact_location)