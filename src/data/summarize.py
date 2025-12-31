import pandas as pd
def summarize_df(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"rows={len(df)}, cols={df.shape[1]}")

    target_cols = ["vacc_h1n1_f", "vacc_seas_f"]
    targets = [c for c in target_cols if c in df.columns]

    if targets:
        lines.append("\nTarget distribution:")
        for t in targets:
            lines.append(f"- {t}: positive_rate={df[t].mean():.3f}")

    miss = df.isna().mean().sort_values(ascending=False).head(10)
    lines.append("\nTop missing columns:")
    lines.append(miss.to_string())

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = [c for c in df.columns if c not in num_cols]
    lines.append(f"\nnum_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

    return "\n".join(lines)
