from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGETS = ["vacc_h1n1_f", "vacc_seas_f"]

ARTIFACT_DIR = Path("data/artifacts")
OUTPUT_DIR = Path("data/outputs")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def split_num_cat_cols(df: pd.DataFrame, targets: List[str]) -> Tuple[List[str], List[str]]:
    feat_cols = [c for c in df.columns if c not in targets]
    X = df[feat_cols]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]
    return num_cols, cat_cols


def preprocess_fit_transform(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    stats: Dict[str, float] = {}
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()

    # numeric -> median impute
    for c in num_cols:
        med = float(np.nanmedian(X_tr2[c].values))
        stats[c] = med
        X_tr2[c] = X_tr2[c].fillna(med)
        X_va2[c] = X_va2[c].fillna(med)

    # categorical -> string + missing token
    for c in cat_cols:
        X_tr2[c] = X_tr2[c].astype("string").fillna("missing")
        X_va2[c] = X_va2[c].astype("string").fillna("missing")

    return X_tr2, X_va2, stats


def preprocess_apply(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    stats: Dict[str, float],
) -> pd.DataFrame:
    X2 = X.copy()
    for c in num_cols:
        X2[c] = X2[c].fillna(stats[c])
    for c in cat_cols:
        X2[c] = X2[c].astype("string").fillna("missing")
    return X2


@dataclass
class FoldResult:
    f1: float
    auc: float
    thr: float


def fit_predict_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    cat_cols: List[str],
    seed: int,
) -> np.ndarray:
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.05,
        iterations=2000,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        l2_leaf_reg=4.0,
        min_data_in_leaf=10,
        subsample=0.8,
        rsm=0.8,
    )

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va, cat_features=cat_idx)

    model.fit(tr_pool)
    return model.predict_proba(va_pool)[:, 1]


def fit_predict_lgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    cat_cols: List[str],
    seed: int,
) -> np.ndarray:
    # LightGBM can handle pandas categorical
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    for c in cat_cols:
        X_tr2[c] = X_tr2[c].astype("category")
        X_va2[c] = X_va2[c].astype("category")

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_samples=10,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_tr2,
        y_tr,
    )
    return model.predict_proba(X_va2)[:, 1]


def cv_min_ensemble(
    df: pd.DataFrame,
    targets: List[str] = TARGETS,
    n_splits: int = 5,
    seed: int = 42,
    alphas: List[float] = [0.5, 0.6, 0.7],
) -> Dict[str, dict]:
    num_cols, cat_cols = split_num_cat_cols(df, targets)
    feat_cols = [c for c in df.columns if c not in targets]

    results: Dict[str, dict] = {}

    for target in targets:
        y = df[target].astype(int).to_numpy()
        X = df[feat_cols]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_cb = np.zeros(len(df), dtype=float)
        oof_lgb = np.zeros(len(df), dtype=float)
        oof_ens = {a: np.zeros(len(df), dtype=float) for a in alphas}

        fold_res_cb: List[FoldResult] = []
        fold_res_lgb: List[FoldResult] = []
        fold_res_ens: Dict[float, List[FoldResult]] = {a: [] for a in alphas}

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            X_tr2, X_va2, stats = preprocess_fit_transform(X_tr, X_va, num_cols, cat_cols)

            # CB
            p_cb = fit_predict_catboost(X_tr2, y_tr, X_va2, cat_cols, seed + fold)
            oof_cb[va_idx] = p_cb
            thr_cb, f1_cb = best_f1_threshold(y_va, p_cb)
            auc_cb = roc_auc_score(y_va, p_cb)
            fold_res_cb.append(FoldResult(f1=f1_cb, auc=float(auc_cb), thr=thr_cb))

            # LGB
            p_lgb = fit_predict_lgbm(X_tr2, y_tr, X_va2, cat_cols, seed + 1000 + fold)
            oof_lgb[va_idx] = p_lgb
            thr_lgb, f1_lgb = best_f1_threshold(y_va, p_lgb)
            auc_lgb = roc_auc_score(y_va, p_lgb)
            fold_res_lgb.append(FoldResult(f1=f1_lgb, auc=float(auc_lgb), thr=thr_lgb))

            # Ensemble (soft voting)
            for a in alphas:
                p = a * p_cb + (1 - a) * p_lgb
                oof_ens[a][va_idx] = p
                thr_e, f1_e = best_f1_threshold(y_va, p)
                auc_e = roc_auc_score(y_va, p)
                fold_res_ens[a].append(FoldResult(f1=f1_e, auc=float(auc_e), thr=thr_e))

            # fold print (최소 정보)
            print(
                f"[{target}] fold{fold} "
                f"CB thr={thr_cb:.3f} F1={f1_cb:.4f} AUC={auc_cb:.5f} | "
                f"LGB thr={thr_lgb:.3f} F1={f1_lgb:.4f} AUC={auc_lgb:.5f}"
            )

        # summarize
        def summarize_fold(fr: List[FoldResult]) -> dict:
            f1s = [x.f1 for x in fr]
            aucs = [x.auc for x in fr]
            thrs = [x.thr for x in fr]
            return {
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "fold_f1": f1s,
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "fold_auc": aucs,
                "mean_thr": float(np.mean(thrs)),
                "std_thr": float(np.std(thrs)),
                "fold_thr": thrs,
            }

        cb_sum = summarize_fold(fold_res_cb)
        lgb_sum = summarize_fold(fold_res_lgb)

        ens_summaries: Dict[float, dict] = {}
        for a in alphas:
            ens_summaries[a] = summarize_fold(fold_res_ens[a])

        # pick best ensemble alpha by mean_f1
        best_alpha = max(alphas, key=lambda a: ens_summaries[a]["mean_f1"])
        best_ens = ens_summaries[best_alpha]

        results[target] = {
            "catboost": cb_sum,
            "lightgbm": lgb_sum,
            "ensemble": {
                "best_alpha": float(best_alpha),
                "summary_by_alpha": ens_summaries,
                "best_summary": best_ens,
            },
        }

        print(
            f"\n[{target}] SUMMARY "
            f"CB meanF1={cb_sum['mean_f1']:.4f} | "
            f"LGB meanF1={lgb_sum['mean_f1']:.4f} | "
            f"ENS(best a={best_alpha}) meanF1={best_ens['mean_f1']:.4f}\n"
        )

    return results


def train_final_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_results: Dict[str, dict],
    targets: List[str] = TARGETS,
    seed: int = 42,
) -> pd.DataFrame:
    num_cols, cat_cols = split_num_cat_cols(train_df, targets)
    feat_cols = [c for c in train_df.columns if c not in targets]

    X_tr = train_df[feat_cols]
    X_te = test_df[feat_cols]

    out = pd.DataFrame(index=test_df.index)

    for target in targets:
        y = train_df[target].astype(int).to_numpy()
        best_alpha = float(cv_results[target]["ensemble"]["best_alpha"])
        best_thr = float(cv_results[target]["ensemble"]["best_summary"]["mean_thr"])

        # 전처리 통계는 train 전체로 재학습
        X_tr2, X_te2, stats = preprocess_fit_transform(X_tr, X_te, num_cols, cat_cols)

        # --- CatBoost train full ---
        cat_idx = [X_tr2.columns.get_loc(c) for c in cat_cols]
        cb = CatBoostClassifier(
            depth=7,
            learning_rate=0.05,
            iterations=2500,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            l2_leaf_reg=4.0,
            min_data_in_leaf=10,
            subsample=0.8,
            rsm=0.8,
        )
        cb.fit(Pool(X_tr2, y, cat_features=cat_idx))
        p_cb = cb.predict_proba(Pool(X_te2, cat_features=cat_idx))[:, 1]

        # --- LGB train full ---
        X_tr_lgb = X_tr2.copy()
        X_te_lgb = X_te2.copy()
        for c in cat_cols:
            X_tr_lgb[c] = X_tr_lgb[c].astype("category")
            X_te_lgb[c] = X_te_lgb[c].astype("category")

        lgb = LGBMClassifier(
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_samples=10,
            random_state=seed,
            n_jobs=-1,
        )
        lgb.fit(X_tr_lgb, y)
        p_lgb = lgb.predict_proba(X_te_lgb)[:, 1]

        # --- Ensemble proba ---
        p_ens = best_alpha * p_cb + (1 - best_alpha) * p_lgb
        out[target] = p_ens

        # artifacts save (최소)
        cb_path = ARTIFACT_DIR / f"final_cb_{target}.cbm"
        lgb_path = ARTIFACT_DIR / f"final_lgb_{target}.txt"
        meta_path = ARTIFACT_DIR / f"final_meta_{target}.json"

        cb.save_model(str(cb_path))
        lgb.booster_.save_model(str(lgb_path))
        meta_path.write_text(
            json.dumps(
                {
                    "target": target,
                    "best_alpha": best_alpha,
                    "best_thr_mean_over_folds": best_thr,
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[FINAL {target}] best_alpha={best_alpha:.3f} "
            f"(fold-mean thr≈{best_thr:.3f}) "
            f"saved: {cb_path.name}, {lgb_path.name}, {meta_path.name}"
        )

    return out


def save_submission(pred_df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)
    print(f"Saved submission-like file -> {path.as_posix()}")
