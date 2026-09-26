import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

path = mlflow.artifacts.download_artifacts(
    artifact_uri='models:/m-a9a7c2412f5b401481f178f93567a5bd'
)
print("Downloaded to:", path)