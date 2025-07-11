from abc import ABC, abstractmethod
from datetime import datetime

import mlflow


class BaseLogger(ABC):
    @abstractmethod
    def log_params(self, params: dict): ...
    @abstractmethod
    def log_metric(self, name: str, value: float): ...
    @abstractmethod
    def log_artifact(self, filepath: str, artifact_path: str=None): ...

class MLflowLogger(BaseLogger):
    def __init__(self, exp_name: str):
        mlflow.set_experiment(exp_name)
        # Use current timestamp as run name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run = mlflow.start_run(run_name=timestamp)
        mlflow.sklearn.autolog()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, name, value):
        mlflow.log_metric(name, value)

    def log_artifact(self, filepath, artifact_path=None):
        mlflow.log_artifact(filepath, artifact_path)

    def log_model(self, estimator):
        mlflow.set_tag('model', estimator.__class__.__name__)

    def __del__(self):
        mlflow.end_run()
