import pandas as pd

RAW_DIR = "data/raw"

def load_train():
    X = pd.read_csv(f"{RAW_DIR}/train.csv")
    y = pd.read_csv(f"{RAW_DIR}/train_labels.csv")

    # row order 기준 결합
    df = pd.concat([X, y], axis=1)
    return df

def load_test():
    return pd.read_csv(f"{RAW_DIR}/test.csv")

def split_xy(df):
    y_cols = ["vacc_h1n1_f", "vacc_seas_f"]

    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Target columns missing: {missing}")

    X = df.drop(columns=y_cols)
    y = df[y_cols].copy()

    return X, y
