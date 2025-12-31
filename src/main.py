from data.load import load_train
from models.baseline import cv_metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = load_train()

for t in ["vacc_h1n1_f", "vacc_seas_f"]:
    res = cv_metrics(df, t, n_splits=5, seed=42)
    print(res)
