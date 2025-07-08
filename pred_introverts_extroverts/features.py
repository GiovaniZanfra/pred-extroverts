# Updated feature.py
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Paths
paths_cfg = config['paths']
INTERIM_PATH = Path(paths_cfg['interim'])
PROCESSED_PATH = Path(paths_cfg['processed'])

# Feature steps
fe_cfg = config['feature_engineering']

# 1. flag missing
def flag_missing(df):
    if not fe_cfg['flag_missing']['enable']:
        return df
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f'is_null_{col}'] = df[col].isna().astype(int)
    df['any_missing'] = df.isna().any(axis=1).astype(int)
    return df

# 2. composite mean
def add_composite_mean(df):
    cfg = fe_cfg['add_composite_mean']
    if not cfg['enable']:
        return df
    cols = cfg['cols']
    df = df.copy()
    df['social_score_mean'] = df[cols].mean(axis=1)
    return df

# 3. composite PCA
def add_composite_pca(df):
    cfg = fe_cfg['add_composite_pca']
    if not cfg['enable']:
        return df
    cols = cfg['cols']
    df = df.copy()
    if cfg.get('scale_before', False):
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cols])
    else:
        X = df[cols].values
    pca = PCA(n_components=cfg['n_components'])
    comp = pca.fit_transform(X)
    df['social_score_pca'] = comp.flatten()
    print(f"PCA var: {pca.explained_variance_ratio_[0]:.2%}")
    return df

# 4. numeric combos
def numeric_combinations(df):
    cfg = fe_cfg['numeric_combinations']
    if not cfg['enable']:
        return df
    cols = cfg.get('cols', None)
    ops = cfg['operations']
    eps = cfg.get('eps', 1e-8)
    df = df.copy()
    num_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if not c.startswith('is_null') and c!='any_missing']
    for c1, c2 in combinations(num_cols, 2):
        if 'product' in ops:
            df[f'{c1}_x_{c2}'] = df[c1]*df[c2]
        if 'diff' in ops:
            df[f'{c1}_minus_{c2}'] = df[c1]-df[c2]
        if 'ratio' in ops:
            df[f'{c1}_div_{c2}'] = df[c1]/(df[c2]+eps)
    return df

# 5. categorical combos
def categorical_combinations(df):
    cfg = fe_cfg['categorical_combinations']
    if not cfg['enable']:
        return df
    sep = cfg['sep']
    cols = cfg.get('cols', None)
    df = df.copy()
    cat_cols = cols or df.select_dtypes(include=['object','category']).columns.tolist()
    if 'Personality' in cat_cols:
        cat_cols.remove('Personality')
    for c1, c2 in combinations(cat_cols, 2):
        df[f'{c1}{sep}{c2}'] = df[c1].astype(str)+sep+df[c2].astype(str)
    return df

# pipeline
STEP_FUNCS = {
    'flag_missing': flag_missing,
    'add_composite_mean': add_composite_mean,
    'add_composite_pca': add_composite_pca,
    'numeric_combinations': numeric_combinations,
    'categorical_combinations': categorical_combinations,
}

def process_file(fp: Path) -> pd.DataFrame:
    print(f"Proc {fp.name}...")
    df = pd.read_csv(fp)
    for step in fe_cfg['steps']:
        df = STEP_FUNCS[step](df)
    return df

# main
def main():
    if not fe_cfg['enable']:    
        print("feature.py desativado.")
        return
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    for fn in ['train.csv','test.csv']:
        df = process_file(INTERIM_PATH / fn)
        df.to_csv(PROCESSED_PATH / fn, index=False)
    print(f"Salvo em {PROCESSED_PATH}.")

if __name__ == "__main__":
    main()
