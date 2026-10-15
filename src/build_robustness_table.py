"""
build_robustness_table.py
--------------------------
1. Loads the 4 robustness CSVs from data/robustness/.
2. Reports duplicate model names per CSV, then merges on
   ["model", "img_size", "crop_pct"] (asserts <= 1556 rows).
3. Drops test_ models and models with clean_top1 < 60.
4. Adds pretrain_tag and extra_data columns.
5. Fits OLS logit regressions and stores residuals.
6. Saves results/robustness_table.csv.
7. Saves 3 scatter plots colored by extra_data.
8. Prints group means and top/bottom 10 by sketch_resid.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "robustness"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

MERGE_KEYS = ["model", "img_size", "crop_pct"]

EXTRA_DATA_TOKENS = {
    "clip", "laion", "openai", "siglip", "webli", "ig1b",
    "swsl", "wsl", "ssl", "yfcc", "in22k", "in21k", "in12k",
    "jft", "m38m", "m30m", "mim", "dfn", "lvd",
}

# ---------------------------------------------------------------------------
# 1. Load CSVs and report duplicates
# ---------------------------------------------------------------------------
files = {
    "imagenet":  "results-imagenet.csv",
    "sketch":    "results-sketch.csv",
    "imagenet_a": "results-imagenet-a.csv",
    "imagenet_r": "results-imagenet-r.csv",
}

raw = {}
print("=" * 60)
print("DUPLICATE MODEL NAMES PER CSV")
print("=" * 60)
for key, fname in files.items():
    df = pd.read_csv(DATA_DIR / fname)
    dup_count = df["model"].duplicated(keep=False).sum()
    n_unique   = df["model"].nunique()
    print(f"  {fname}: {len(df)} rows, {n_unique} unique model names, "
          f"{dup_count} rows involved in duplicates")
    raw[key] = df

# ---------------------------------------------------------------------------
# 2. Merge on ["model", "img_size", "crop_pct"]
# ---------------------------------------------------------------------------
def pick(df, top1_alias, extra_cols=None):
    keep = MERGE_KEYS + ["top1"]
    if extra_cols:
        keep += extra_cols
    return df[keep].rename(columns={"top1": top1_alias})

df = (
    pick(raw["imagenet"], "clean_top1", extra_cols=["param_count"])
    .merge(pick(raw["sketch"],    "sketch_top1"),     on=MERGE_KEYS, how="inner")
    .merge(pick(raw["imagenet_a"], "imagenet_a_top1"), on=MERGE_KEYS, how="inner")
    .merge(pick(raw["imagenet_r"], "imagenet_r_top1"), on=MERGE_KEYS, how="inner")
)

print(f"\nMerged on {MERGE_KEYS}: {df.shape[0]} rows")
assert df.shape[0] <= 1556, f"Expected <= 1556 rows, got {df.shape[0]}"
print("Assertion passed: <= 1556 rows.")

# ---------------------------------------------------------------------------
# 3. Drop test_ models and clean_top1 < 60
# ---------------------------------------------------------------------------
n_before = len(df)
df = df[~df["model"].str.startswith("test_")]
df = df[df["clean_top1"] >= 60.0].copy()
print(f"\nAfter filtering: {n_before} -> {len(df)} rows "
      f"(dropped {n_before - len(df)} rows)")

# ---------------------------------------------------------------------------
# 4. pretrain_tag and extra_data columns
# ---------------------------------------------------------------------------
def get_pretrain_tag(model_name: str) -> str:
    parts = model_name.split(".", 1)
    return parts[1] if len(parts) > 1 else ""

df["pretrain_tag"] = df["model"].apply(get_pretrain_tag)

def has_extra_data(tag: str) -> bool:
    tag_lower = tag.lower()
    return any(tok in tag_lower for tok in EXTRA_DATA_TOKENS)

df["extra_data"] = df["pretrain_tag"].apply(has_extra_data)
print(f"\nextra_data breakdown:")
print(df["extra_data"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 5. Logit transform
# ---------------------------------------------------------------------------
acc_cols = ["clean_top1", "sketch_top1", "imagenet_a_top1", "imagenet_r_top1"]
for col in acc_cols:
    p = (df[col] / 100.0).clip(1e-6, 1 - 1e-6)
    df[f"logit_{col}"] = np.log(p / (1 - p))

# ---------------------------------------------------------------------------
# 6. OLS regressions and residuals
# ---------------------------------------------------------------------------
benchmarks = [
    ("sketch_top1",      "sketch_resid",  "ImageNet-Sketch"),
    ("imagenet_a_top1",  "a_resid",       "ImageNet-A"),
    ("imagenet_r_top1",  "r_resid",       "ImageNet-R"),
]

fitted_lines = {}

print("\n" + "=" * 60)
print("REGRESSION: logit(rob) ~ logit(clean)")
print("=" * 60)
for rob_col, resid_col, label in benchmarks:
    x = df["logit_clean_top1"].values
    y = df[f"logit_{rob_col}"].values
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    df[resid_col] = np.where(mask, y - (slope * x + intercept), np.nan)
    fitted_lines[rob_col] = (slope, intercept, r**2, label)
    print(f"  {label}: slope={slope:.4f}, intercept={intercept:.4f}, "
          f"R2={r**2:.4f}")

# ---------------------------------------------------------------------------
# 7. Save table
# ---------------------------------------------------------------------------
out_csv = RES_DIR / "robustness_table.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved table -> {out_csv}  ({df.shape[0]} rows, {df.shape[1]} cols)")

# ---------------------------------------------------------------------------
# 8. Scatter plots colored by extra_data
# ---------------------------------------------------------------------------
COLORS = {True: "#E84040", False: "#4C72B0"}   # red=extra data, blue=standard
LABELS = {True: "Extra-data pretraining", False: "Standard pretraining"}
ALPHA  = 0.45
SIZE   = 14

for rob_col, resid_col, label in benchmarks:
    slope, intercept, r2, _ = fitted_lines[rob_col]
    x_all = df["logit_clean_top1"].values
    y_all = df[f"logit_{rob_col}"].values

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for flag in [False, True]:
        mask = df["extra_data"] == flag
        ax.scatter(
            x_all[mask], y_all[mask],
            alpha=ALPHA, s=SIZE,
            color=COLORS[flag],
            label=f"{LABELS[flag]} (n={mask.sum()})",
            zorder=2,
        )

    # OLS line
    x_line = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 300)
    ax.plot(x_line, slope * x_line + intercept,
            color="black", lw=1.8, ls="--",
            label=f"OLS fit  R²={r2:.3f}", zorder=3)

    ax.set_xlabel("logit(Clean ImageNet Top-1)", fontsize=11)
    ax.set_ylabel(f"logit({label} Top-1)", fontsize=11)
    ax.set_title(f"Clean vs {label} (logit scale)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    safe = label.lower().replace(" ", "_").replace("-", "_")
    png_path = RES_DIR / f"scatter_{safe}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot  -> {png_path}")

# ---------------------------------------------------------------------------
# 9. Group means: extra_data True vs False
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MEAN RESIDUALS BY extra_data")
print("=" * 60)
group_means = (
    df.groupby("extra_data")[["sketch_resid", "r_resid"]]
    .mean()
    .rename(index={True: "extra_data=True", False: "extra_data=False"})
)
print(group_means.to_string(float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# 10. Top / Bottom 10 by sketch_resid (include img_size)
# ---------------------------------------------------------------------------
df_valid = df.dropna(subset=["sketch_resid"]).sort_values(
    "sketch_resid", ascending=False
)
display_cols = ["model", "img_size", "param_count",
                "clean_top1", "sketch_top1", "sketch_resid", "extra_data"]

print("\n" + "=" * 70)
print("TOP 10 by sketch_resid")
print("=" * 70)
print(df_valid.head(10)[display_cols].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"
))

print("\n" + "=" * 70)
print("BOTTOM 10 by sketch_resid")
print("=" * 70)
print(df_valid.tail(10)[display_cols].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"
))

print("\nDone.")
