from mlflow import MlflowClient

c = MlflowClient('http://127.0.0.1:5000')
mv = c.create_model_version(
    name='CustomerChurnModel',
    source='models:/m-a9a7c2412f5b401481f178f93567a5bd',
    model_id='m-a9a7c2412f5b401481f178f93567a5bd'
)
c.set_registered_model_alias('CustomerChurnModel', 'champion', mv.version)
print("New champion version:", mv.version)