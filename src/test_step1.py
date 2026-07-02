import json
import pandas as pd
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features

print("Loading candidates...")
candidates = []
with open("../data/candidates.jsonl") as f:
    for line in f:
        candidates.append(json.loads(line.strip()))

print(f"Loaded {len(candidates):,} candidates")

print("Extracting features...")
start = time.time()

rows = []
for i, c in enumerate(candidates):
    rows.append(extract_features(c))
    if (i + 1) % 10000 == 0:
        print(f"  processed {i+1:,} / {len(candidates):,}...")

elapsed = time.time() - start
print(f"Done in {elapsed:.1f} seconds")

# Save to parquet
os.makedirs("../artifacts", exist_ok=True)
df = pd.DataFrame(rows)
df.to_parquet("../artifacts/features_full.parquet", index=False)
print(f"\nSaved to artifacts/features_full.parquet")
print(f"Shape: {df.shape}")

# Summary stats
print(f"\n--- FLAG COUNTS ---")
flag_cols = [
    "flag_consulting_only", "flag_title_chaser",
    "flag_cv_speech_robotics", "flag_honeypot", "flag_langchain_only"
]
for col in flag_cols:
    print(f"  {col}: {df[col].sum():,}")

print(f"\n--- SCORE DISTRIBUTION ---")
print(f"  must_have_ratio > 0.5  : {(df['must_have_ratio'] > 0.5).sum():,}")
print(f"  must_have_ratio > 0.75 : {(df['must_have_ratio'] > 0.75).sum():,}")
print(f"  must_have_ratio == 1.0 : {(df['must_have_ratio'] == 1.0).sum():,}")
print(f"  production_evidence    : {df['production_evidence'].sum():,}")

print(f"\n--- TOP 10 CANDIDATES BY MUST-HAVE SCORE ---")
top = df.nlargest(10, "must_have_ratio")[
    ["candidate_id", "current_title", "years_of_experience",
     "must_have_ratio", "nice_ratio", "flag_count", "production_evidence"]
]
print(top.to_string(index=False))