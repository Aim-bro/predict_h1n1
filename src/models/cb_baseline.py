import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from catboost import CatBoostClassifier


TARGETS = ["vacc_h1n1_f", "vacc_seas_f"]

MISSING_FLAG_COLS = [
    "employment_industry",
    "employment_occupation",
    "health_insurance",
    "education_comp",
]


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    for c in MISSING_FLAG_COLS:
        if c in X2.columns:
            X2[f"{c}_missing"] = X2[c].isna().astype(int)
    return X2


def best_f1_threshold_2stage(y_true: pd.Series, y_prob: np.ndarray):
    y_true_np = np.asarray(y_true).astype(int)

    def _scan(ths):
        best_t = 0.5
        best_f1 = -1.0
        for t in ths:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true_np, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t, float(best_f1)

    coarse = np.arange(0.20, 0.81, 0.02)
    t0, _ = _scan(coarse)

    lo = max(0.01, t0 - 0.06)
    hi = min(0.99, t0 + 0.06)
    fine = np.arange(lo, hi + 1e-9, 0.005)

    return _scan(fine)


def _cat_cols(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    return [c for c in X.columns if c not in num_cols]


def _prep_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """
    CatBoost는 문자열/카테고리 입력이 자연스러움
    - 범주형: NaN -> "missing", dtype object 유지
    - 수치형: 그대로(결측은 CatBoost가 처리)
    """
    X2 = X.copy()

    for c in cat_cols:
        X2[c] = X2[c].astype("object").where(~X2[c].isna(), "missing")

    return X2


def cv_f1_with_inner_threshold_catboost(
    df: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    seed: int = 42,
    depth : int = 6,
) -> dict:
    if target not in df.columns:
        raise ValueError(f"target not found: {target}")

    y = df[target].astype(int)
    X = df.drop(columns=TARGETS)
    X = add_missing_flags(X)

    cat_cols = _cat_cols(X)
    cat_features = [X.columns.get_loc(c) for c in cat_cols]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_f1 = []
    fold_auc = []
    fold_thr = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        tr2_idx, tune_idx = train_test_split(
            np.arange(len(X_tr)),
            test_size=0.2,
            random_state=seed + fold,
            stratify=y_tr,
        )
        X_tr2, y_tr2 = X_tr.iloc[tr2_idx], y_tr.iloc[tr2_idx]
        X_tune, y_tune = X_tr.iloc[tune_idx], y_tr.iloc[tune_idx]

        # CatBoost 입력 전처리
        X_tr2_p = _prep_for_catboost(X_tr2, cat_cols)
        X_tune_p = _prep_for_catboost(X_tune, cat_cols)
        X_va_p = _prep_for_catboost(X_va, cat_cols)

        model = CatBoostClassifier(
            iterations=8000,
            learning_rate=0.03,
            depth=depth,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed + fold,
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(
            X_tr2_p,
            y_tr2,
            cat_features=cat_features,
            eval_set=(X_tune_p, y_tune),
            use_best_model=True,
        )

        tune_proba = model.predict_proba(X_tune_p)[:, 1]
        t, _ = best_f1_threshold_2stage(y_tune, tune_proba)

        va_proba = model.predict_proba(X_va_p)[:, 1]
        auc = roc_auc_score(y_va, va_proba)
        f1 = f1_score(y_va, (va_proba >= t).astype(int))

        fold_thr.append(float(t))
        fold_auc.append(float(auc))
        fold_f1.append(float(f1))

        print(f"[{target}] fold{fold} thr={t:.3f} AUC={auc:.5f} F1={f1:.4f}")

    return {
        "target": target,
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "fold_f1": fold_f1,
        "mean_auc": float(np.mean(fold_auc)),
        "std_auc": float(np.std(fold_auc)),
        "fold_auc": fold_auc,
        "mean_thr": float(np.mean(fold_thr)),
        "std_thr": float(np.std(fold_thr)),
        "fold_thr": fold_thr,
    }
