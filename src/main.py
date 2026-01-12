from data.load import load_train
from models.baseline import cv_f1_with_inner_threshold
from models.lgb_baseline import cv_f1_with_inner_threshold_lgbm
from models.cb_baseline import cv_f1_with_inner_threshold_catboost
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# base line 
# res = cv_f1_with_inner_threshold(df, "vacc_h1n1_f", n_splits=5, seed=42)
# lgb base line
# res = cv_f1_with_inner_threshold_lgbm(df, "vacc_h1n1_f", n_splits=5, seed=42)
##################################
# df = load_train()
# for d in [6, 7, 8]:
#     res = cv_f1_with_inner_threshold_catboost(
#         df,
#         "vacc_h1n1_f",
#         n_splits=5,
#         seed=42,
#         depth=d,
#     )
#     print(f"DEPTH={d}", res["mean_f1"], res["std_f1"])

import time

from data.load import load_train, load_test 
from models.ensemble import cv_min_ensemble, train_final_and_predict, save_submission

def main():
    t0 = time.time()

    train_df = load_train()
    test_df = load_test()

    cv_results = cv_min_ensemble(
        train_df,
        n_splits=5,
        seed=42,
        alphas=[0.5, 0.6, 0.7],
    )

    # 최종 선택된 앙상블로 test 예측 생성 + 모델 저장
    pred_test = train_final_and_predict(train_df, test_df, cv_results, seed=42)

    # 제출 직전 형태의 파일(확률) 생성
    save_submission(pred_test, "data/outputs/submission_ensemble.csv")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed/60:.2f} minutes ({elapsed:.1f} sec)")

if __name__ == "__main__":
    main()
