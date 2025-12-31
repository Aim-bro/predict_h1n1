import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


TARGETS = ["vacc_h1n1_f", "vacc_seas_f"]


def _make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray):
    """
    OOF 확률(y_prob) 기반으로 F1을 최대화하는 threshold 탐색
    """
    thresholds = np.linspace(0.05, 0.95, 91)

    best_t = 0.5
    best_f1 = -1.0

    y_true_np = np.asarray(y_true).astype(int)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true_np, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, float(best_f1)


def cv_metrics(df: pd.DataFrame, target: str, n_splits: int = 5, seed: int = 42) -> dict:
    """
    - fold별 AUC 출력
    - OOF 기반 AUC 계산
    - OOF 기반 best threshold 탐색 후 F1 계산
    """
    if target not in df.columns:
        raise ValueError(f"target not found: {target}")

    y = df[target].astype(int)
    X = df.drop(columns=TARGETS)

    preprocess = _make_preprocess(X)

    # n_jobs 경고 나오는 버전이 있어서 빼는 게 깔끔
    clf = LogisticRegression(max_iter=3000)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(df), dtype=float)
    fold_auc = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]

        oof[va_idx] = proba

        auc = roc_auc_score(y_va, proba)
        fold_auc.append(float(auc))
        print(f"[{target}] fold{fold} AUC={auc:.5f}")

    oof_auc = float(roc_auc_score(y, oof))
    t, best_f1 = best_f1_threshold(y, oof)

    print(f"[{target}] OOF AUC={oof_auc:.5f}")
    print(f"[{target}] best_threshold={t:.3f}, OOF F1={best_f1:.4f}")

    return {
        "target": target,
        "fold_auc": fold_auc,
        "mean_auc": float(np.mean(fold_auc)),
        "std_auc": float(np.std(fold_auc)),
        "oof_auc": oof_auc,
        "best_threshold": t,
        "oof_f1": best_f1,
    }
