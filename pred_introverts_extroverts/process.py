# Updated process.py with imputation
import pickle
from pathlib import Path

import pandas as pd
import yaml

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Paths from config
def get_paths():
    p = config['paths']
    return Path(p['raw']), Path(p['interim']), Path(p.get('cv_indices', ''))

# Binomial encoding
def encode_binomial_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = config['process']['encode_binomial']
    if not (config['process']['enable'] and cfg['enable']):
        return df
    bin_map = cfg['binomial_map']
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().unique()
        if len(vals) == 2 and all(v in bin_map for v in vals):
            df[col] = df[col].map(bin_map).astype("category")
    return df


def main():
    # Check if process step is enabled
    if not config['process']['enable']:
        print("process.py disabled by config.")
        return

    raw_path, interim_path, _ = get_paths()
    print(f"Loading raw data from {raw_path}...")
    train = pd.read_csv(raw_path / "train.csv")
    test = pd.read_csv(raw_path / "test.csv")

    print("Encoding binomial features...")
    train_enc = encode_binomial_features(train)
    test_enc  = encode_binomial_features(test)

    # Save interim
    interim_path.mkdir(parents=True, exist_ok=True)
    train_enc.to_csv(interim_path / "train.csv", index=False)
    test_enc.to_csv(interim_path / "test.csv", index=False)
    print(f"Processed data saved in {interim_path}.")

if __name__ == "__main__":
    main()
