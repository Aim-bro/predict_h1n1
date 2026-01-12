# H1N1 Vaccination Prediction (F1 Optimization with LLM-assisted Tuning)

## 1. Problem
- Predict H1N1 vaccination adoption
- Evaluation metric: **F1-score**
- Historical top-10 score: ~0.629

## 2. Dataset
- Kaggle: Prediction of H1N1 Vaccination
- Survey-based data with:
  - Many categorical features
  - High missingness
  - Class imbalance

## 3. Baseline
- Logistic Regression
- CV F1 ≈ 0.59

## 4. Tree-based Models
### LightGBM
- logloss + early stopping
- missing flags
- 2-stage threshold optimization
- CV F1 ≈ 0.62

### CatBoost (Main Model)
- Native categorical handling
- Missing-value aware boosting
- Inner-fold threshold optimization
- **CV F1 = 0.6347**

## 5. Threshold Optimization
- 2-stage search:
  - coarse (0.20 ~ 0.80, step 0.02)
  - fine (±0.06, step 0.005)
- Stabilized fold-wise F1

## 6. Tuning
- CatBoost depth tuning in progress (6 / 7 / 8)
- Partial runs completed (full results to be added)

## 7. Ensemble
- Complete depth comparison
- CatBoost + LightGBM ensemble

본 프로젝트에서는 CatBoost를 주력 모델로 사용해 H1N1 및 Seasonal Flu 백신 접종 여부를 예측했다.
5-Fold CV 기준 H1N1 타깃에서 F1 ≈ 0.63을 달성하여 과거 상위 10등 성능을 안정적으로 상회했다.

추가로 LightGBM을 결합한 최소 앙상블(soft voting)을 검증했으며, Seasonal 타깃에서는 소폭의 성능 개선을 확인했다.
반면 H1N1 타깃에서는 단일 CatBoost 모델이 이미 충분히 강력함을 확인하여, 타깃 특성에 따른 모델 선택 전략을 적용했다.
