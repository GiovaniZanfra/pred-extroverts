# train.py
import argparse
from pathlib import Path

import pandas as pd
import yaml

# Transformers
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression

# CV and predictions
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    elif name == 'xgbclassifier':
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown estimator: {name}")


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

    # Build pipeline
    estimator = get_estimator(cfg['estimator'])
    pipe = build_pipeline(cfg['pipeline_steps'], estimator)

    # CV setup (supports stratified)
    cv_cfg = cfg['cv']
    cv_type = cv_cfg.get('type', 'kfold')
    cv_params = {k: v for k, v in cv_cfg.items() if k != 'type'}
    if cv_type == 'stratified':
        cv = StratifiedKFold(**cv_params)
    else:
        cv = KFold(**cv_params)

    # Fit & predict OOF
    oof_preds_num = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict')

    # Decode predictions if binomial_map provided
    bin_cfg = cfg.get('process', {}).get('encode_binomial', {})
    inv_map = None
    if bin_cfg.get('enable') and 'binomial_map' in bin_cfg:
        inv_map = {v: k for k, v in bin_cfg['binomial_map'].items()}
        # decode true labels as well
        y_train_decoded = y_train.map(inv_map)
        oof_preds = pd.Series(oof_preds_num).map(inv_map)
    else:
        y_train_decoded = y_train
        oof_preds = pd.Series(oof_preds_num)

    oof_df = pd.DataFrame({
        id_col: train_df[id_col],
        'oof_pred': oof_preds,
        'y_true': y_train_decoded
    })
    oof_df.to_csv(cfg['output']['oof'], index=False)

    # Train full and predict test
    pipe.fit(X_train, y_train)
    test_preds_num = pipe.predict(X_test)
    if inv_map:
        test_preds = pd.Series(test_preds_num).map(inv_map)
    else:
        test_preds = pd.Series(test_preds_num)

    test_out = pd.DataFrame({
        id_col: test_df[id_col],
        'prediction': test_preds
    })
    test_out.to_csv(cfg['output']['test'], index=False)

    print(f"OOF predictions saved to {cfg['output']['oof']}")
    print(f"Test predictions saved to {cfg['output']['test']}")


if __name__ == '__main__':
    main()

# Example config.yaml:
# 1. PATHS
# paths:
#   raw: "data/raw"
#   interim: "data/interim"
#   processed: "data/processed"
#   cv_indices: "data/processed/cv_idx.pkl"
#
# data:
#   train: "data/processed/train.csv"
#   test: "data/processed/test.csv"
#   target_col: Personality
#   id_col: id
#
# 2. PROCESSAMENTO BINÁRIO
# process:
#   enable: true
#   encode_binomial:
#     enable: true
#     binomial_map:
#       "Yes": 1
#       "No": 0
#       "True": 1
#       "False": 0
#       Extrovert: 1
#       Introvert: 0
#
# 3. FEATURE ENGINEERING (ignored here)
# feature_engineering:
#   enable: true
#
# 4. CROSS-VALIDATION
# cv:
#   type: stratified    # 'kfold' or 'stratified'
#   n_splits: 5
#   shuffle: True
#   random_state: 42
#
# pipeline_steps:
#   - name: imputation
#     params:
#       strategy: median
#   - name: standardscaler
#     params: {}
#
# estimator:
#   name: logistic_regression   # or xgbclassifier
#   params:
#     C: 1.0
#     penalty: l2
#     max_iter: 100
#
# output:
#   oof: output/oof_preds.csv
#   test: output/test_preds.csv