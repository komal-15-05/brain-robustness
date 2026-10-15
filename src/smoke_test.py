"""
smoke_test.py
-------------
a) Load resnet50 from timm with pretrained weights, run one random 224x224
   image through it, and print the output shape.
b) Load the 4 robustness CSVs with pandas and print the first 3 rows of each.
"""
import sys
import pathlib
import torch
import timm
import pandas as pd

# ---------------------------------------------------------------------------
# a) timm resnet50 forward pass
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART A: timm resnet50 forward pass")
print("=" * 60)

model = timm.create_model("resnet50", pretrained=True)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

# one random 224x224 RGB image, batch size 1
dummy_input = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_input)

print(f"Input  shape : {tuple(dummy_input.shape)}")
print(f"Output shape : {tuple(output.shape)}")

# ---------------------------------------------------------------------------
# b) Load and display the 4 CSVs
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("PART B: Robustness CSVs (first 3 rows each)")
print("=" * 60)

data_dir = pathlib.Path(__file__).parent.parent / "data" / "robustness"
csv_files = [
    "results-imagenet.csv",
    "results-sketch.csv",
    "results-imagenet-a.csv",
    "results-imagenet-r.csv",
]

for fname in csv_files:
    fpath = data_dir / fname
    print(f"\n--- {fname} ---")
    try:
        df = pd.read_csv(fpath)
        print(f"Shape: {df.shape}")
        pd.set_option("display.max_columns", 8)
        pd.set_option("display.width", 120)
        print(df.head(3).to_string())
    except FileNotFoundError:
        print(f"  ERROR: file not found at {fpath}")
        sys.exit(1)

print("\nSmoke test complete.")
