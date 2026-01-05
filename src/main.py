from data.load import load_train
from models.baseline import cv_f1_with_inner_threshold
from models.lgb_baseline import cv_f1_with_inner_threshold_lgbm
from models.cb_baseline import cv_f1_with_inner_threshold_catboost
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = load_train()
# base line 
# res = cv_f1_with_inner_threshold(df, "vacc_h1n1_f", n_splits=5, seed=42)
# lgb base line
# res = cv_f1_with_inner_threshold_lgbm(df, "vacc_h1n1_f", n_splits=5, seed=42)

for d in [6, 7, 8]:
    res = cv_f1_with_inner_threshold_catboost(
        df,
        "vacc_h1n1_f",
        n_splits=5,
        seed=42,
        depth=d,
    )
    print(f"DEPTH={d}", res["mean_f1"], res["std_f1"])
