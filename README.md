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

## 6. Current Status
- CatBoost depth tuning in progress (6 / 7 / 8)
- Partial runs completed (full results to be added)

## 7. Next Steps
- Complete depth comparison
- CatBoost + LightGBM ensemble
- LLM (Gemini) as experiment-recommendation engine
