# train.py
import argparse
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier

# Evaluator and Logger
from evaluator import Evaluator
from logger import MLflowLogger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# Transformers
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
    SGDOneClassSVM,
)

# CV and predictions
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_pipeline(pipeline_cfg, estimator):
    steps = []
    for step in pipeline_cfg:
        name = step['name']
        params = step.get('params', {})
        if name == 'imputation':
            steps.append((name, SimpleImputer(**params)))
        elif name == 'standardscaler':
            steps.append((name, StandardScaler(**params)))
        else:
            raise ValueError(f"Unknown pipeline step: {name}")
    steps.append(('estimator', estimator))
    return Pipeline(steps)


def get_estimator(est_cfg):
    name = est_cfg['name']
    params = est_cfg.get('params', {})
    if name == 'logistic_regression':
        return LogisticRegression(**params)
    elif name == 'logistic_regression_cv':
        return LogisticRegressionCV(**params)
    elif name == 'xgbclassifier':
        return XGBClassifier(**params)
    elif name == 'random_forest':
        return RandomForestClassifier(**params)
    elif name == 'gradient_boosting':
        return GradientBoostingClassifier(**params)
    elif name == 'linear_svc':
        return LinearSVC(**params)
    elif name == 'nusvc':
        return NuSVC(**params)
    elif name == 'svc':
        return SVC(**params)
    elif name == 'extra_tree':
        return ExtraTreesClassifier(**params)
    elif name == 'decision_tree':
        return DecisionTreeClassifier(**params)
    elif name == 'knn':
        return KNeighborsClassifier(**params)
    elif name == 'nearest_centroid':
        return NearestCentroid(**params)
    elif name == 'radius_neighbors':
        return RadiusNeighborsClassifier(**params)
    elif name == 'passive_aggressive':
        return PassiveAggressiveClassifier(**params)
    elif name == 'perceptron':
        return Perceptron(**params)
    elif name == 'ridge':
        return RidgeClassifier(**params)
    elif name == 'sgd':
        return SGDClassifier(**params)
    elif name == 'oneclass_svm':
        return SGDOneClassSVM(**params)
    elif name == 'catboost':
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown estimator: {name}")(f"Unknown estimator: {name}")


def main():
    cfg = load_config()

    # Load data
    train_df = pd.read_csv(cfg['data']['train'])
    test_df = pd.read_csv(cfg['data']['test'])

    target = cfg['data']['target_col']
    id_col = cfg['data']['id_col']
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target]) if target in test_df else test_df
    id_train = train_df[id_col]
    id_test = test_df[id_col]

    # Build pipeline
    estimator = get_estimator(cfg['estimator'])
    pipe = build_pipeline(cfg['pipeline_steps'], estimator)

    # CV setup (supports stratified)
    cv_cfg = cfg['cv']
    cv_type = cv_cfg.get('type', 'kfold')
    cv_params = {k: v for k, v in cv_cfg.items() if k != 'type'}
    cv = StratifiedKFold(**cv_params) if cv_type == 'stratified' else KFold(**cv_params)

    # Fit & predict OOF
    oof_preds_num = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict')

    # Decode predictions if needed
    bin_cfg = cfg.get('process', {}).get('encode_binomial', {})
    inv_map = None
    if bin_cfg.get('enable') and 'binomial_map' in bin_cfg:
        inv_map = {v: k for k, v in bin_cfg['binomial_map'].items()}
        y_train_decoded = y_train.map(inv_map)
        oof_preds = pd.Series(oof_preds_num).map(inv_map)
    else:
        y_train_decoded = y_train
        oof_preds = pd.Series(oof_preds_num)

    # Identify misclassified IDs
    miscl_mask = oof_preds != y_train_decoded
    misclassified_ids = id_train[miscl_mask]

    # Build outputs
    evaluator = Evaluator(id_train, id_test)
    precision = evaluator.oof_precision(y_train_decoded, oof_preds)
    oof_df = evaluator.build_oof_df(oof_preds, y_train_decoded)
    test_preds_num = pipe.fit(X_train, y_train).predict(X_test)
    test_preds = pd.Series(test_preds_num).map(inv_map) if inv_map else pd.Series(test_preds_num)
    test_df_out = evaluator.build_test_df(test_preds)

    # Save CSVs and misclassified IDs list
    oof_df.to_csv(cfg['output']['oof'], index=False)
    test_df_out.to_csv(cfg['output']['test'], index=False)
    miscl_path = Path(cfg['output']['oof']).parent / 'misclassified_ids.txt'
    with open(miscl_path, 'w') as f:
        for iid in misclassified_ids:
            f.write(f"{iid}\n")

    # Log to MLflow
    logger = MLflowLogger(exp_name=cfg.get('mlflow_experiment', 'default'))
    # Log which estimator was used
    logger.log_params({'estimator_name': cfg['estimator']['name']})
    # Log estimator hyperparameters
    logger.log_params(cfg['estimator'].get('params', {}))
    logger.log_metric('oof_precision', precision)
    logger.log_artifact(cfg['output']['oof'], artifact_path='oof')
    logger.log_artifact(cfg['output']['test'], artifact_path='test')
    logger.log_artifact(str(miscl_path), artifact_path='errors')
    logger.log_model(estimator)
    print(f"OOF predictions saved to {cfg['output']['oof']}")
    print(f"Test predictions saved to {cfg['output']['test']}")
    print(f"Misclassified IDs saved to {miscl_path}")


if __name__ == '__main__':
    main()
