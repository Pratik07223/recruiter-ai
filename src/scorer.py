"""
scorer.py

Step 3: Composite scoring and ranking.

Combines all signals from Step 1 (rule-based features) and Step 2
(semantic similarity) into one final score per candidate.

Score is out of 100 points, broken down as:
  - Must-have evidence  : 40 pts  (most important)
  - Semantic fit        : 20 pts
  - Nice-to-have        : 10 pts
  - Production evidence :  5 pts
  - Experience range    :  5 pts
  - Location            :  5 pts
  - Behavioral signals  :  5 pts
  - Flag penalties      : -8 pts per flag, -25 extra for honeypot

Design principle: interpretable weighted formula, not a black box.
Every point awarded or deducted can be explained in plain English.
"""

import pandas as pd
import numpy as np


# -----------------------------------------------------------------------
# 1. Scoring weights — all in one place so they're easy to tune
# -----------------------------------------------------------------------
WEIGHTS = {
    "must_have":          40,   # max points for must-have category evidence
    "semantic":           20,   # max points for semantic similarity
    "nice_to_have":       10,   # max points for nice-to-have categories
    "production_evidence": 5,   # bonus for real deployed-system language
    "experience_range":    5,   # bonus for being in the 5-9yr sweet spot
    "location_preferred":  5,   # bonus for Pune/Noida
    "location_acceptable": 2,   # smaller bonus for other major cities
    "behavioral":          5,   # max points for availability signals
    "flag_penalty":        8,   # deducted per active flag
    "honeypot_extra":     25,   # extra deduction for honeypot flag
}


# -----------------------------------------------------------------------
# 2. Per-row scoring function
# -----------------------------------------------------------------------
def compute_score(row) -> float:
    score = 0.0

    # --- Must-have evidence (0 to 40 pts) ---
    score += row["must_have_ratio"] * WEIGHTS["must_have"]

    # --- Semantic fit (0 to 20 pts) ---
    # Normalize semantic_score from [0.05, 0.71] range to [0, 1]
    # We observed min=0.055, max=0.713 in our dataset
    sem_normalized = (row["semantic_score"] - 0.05) / (0.713 - 0.05)
    sem_normalized = max(0.0, min(1.0, sem_normalized))
    score += sem_normalized * WEIGHTS["semantic"]

    # --- Nice-to-have (0 to 10 pts) ---
    score += row["nice_ratio"] * WEIGHTS["nice_to_have"]

    # --- Production evidence (0 or 5 pts) ---
    if row["production_evidence"]:
        score += WEIGHTS["production_evidence"]

    # --- Experience range (0 or 5 pts) ---
    if row["exp_in_range"]:
        score += WEIGHTS["experience_range"]

    # --- Location (0, 2, or 5 pts) ---
    if row["preferred_location"]:
        score += WEIGHTS["location_preferred"]
    elif row["acceptable_location"]:
        score += WEIGHTS["location_acceptable"]

    # --- Behavioral availability (0 to 5 pts) ---
    # response_rate is already 0-1 scale (0.02 to 0.95)
    # profile_completeness is 0-100 scale — divide by 100 to normalize
    # open_to_work is False for all candidates in this dataset — skip it
    response_rate        = float(row.get("response_rate", 0) or 0)
    profile_completeness = float(row.get("profile_completeness", 0) or 0) / 100.0
    actively_applying    = float(bool(row.get("actively_applying", False)))

    availability = (
        response_rate        * 0.50 +
        profile_completeness * 0.35 +
        actively_applying    * 0.15
    )
    score += availability * WEIGHTS["behavioral"]

    # --- Flag penalties ---
    flag_count = int(row.get("flag_count", 0))
    score -= flag_count * WEIGHTS["flag_penalty"]

    if row.get("flag_honeypot", False):
        score -= WEIGHTS["honeypot_extra"]  # extra hard penalty

    # Clamp to 0-100
    return round(max(0.0, min(100.0, score)), 4)


# -----------------------------------------------------------------------
# 3. Build human-readable reasoning string from features
#    No LLM needed — templated from the actual computed features.
#    This satisfies the submission's "reasoning" column requirement.
# -----------------------------------------------------------------------
def build_reasoning(row) -> str:
    parts = []

    # Must-have evidence
    if row["score_embeddings_retrieval"] == 2:
        parts.append("career-backed embeddings/retrieval experience")
    elif row["score_embeddings_retrieval"] == 1:
        parts.append("embeddings listed as skill only (unverified)")

    if row["score_vector_db"] == 2:
        parts.append("career-backed vector DB/hybrid search experience")
    elif row["score_vector_db"] == 1:
        parts.append("vector DB listed as skill only")

    if row["score_python"] == 2:
        parts.append("Python backed by career history")

    if row["score_eval_frameworks"] == 2:
        parts.append("evaluation frameworks (NDCG/MRR/MAP) in career history")
    elif row["score_eval_frameworks"] == 1:
        parts.append("evaluation frameworks listed as skill only")

    # Semantic
    if row["semantic_score"] >= 0.60:
        parts.append("strong semantic alignment with JD")
    elif row["semantic_score"] >= 0.45:
        parts.append("moderate semantic alignment with JD")

    # Production
    if row["production_evidence"]:
        parts.append("production deployment evidence found")

    # Experience
    exp = row["years_of_experience"]
    if row["exp_in_range"]:
        parts.append(f"{exp}yrs experience (within 5-9yr target range)")
    else:
        parts.append(f"{exp}yrs experience (outside 5-9yr target range)")

    # Flags
    if row["flag_consulting_only"]:
        parts.append("DISQUALIFIER: consulting-only career")
    if row["flag_title_chaser"]:
        parts.append("WARNING: title-chaser pattern detected")
    if row["flag_cv_speech_robotics"]:
        parts.append("WARNING: CV/speech/robotics domain, limited NLP/IR")
    if row["flag_honeypot"]:
        parts.append("DISQUALIFIER: honeypot/impossible profile detected")
    if row["flag_langchain_only"]:
        parts.append("WARNING: LangChain-only AI experience")

    return "; ".join(parts) if parts else "No strong signals found"


# -----------------------------------------------------------------------
# 4. Main scoring function — loads parquet, scores all rows, saves output
# -----------------------------------------------------------------------
def run_scoring():
    print("Loading features_full.parquet...")
    df = pd.read_parquet("../artifacts/features_full.parquet")
    print(f"Loaded {len(df):,} candidates, {df.shape[1]} features")

    # Check semantic_score exists
    if "semantic_score" not in df.columns:
        print("ERROR: semantic_score column missing.")
        print("Please run embeddings.py first.")
        return

    # --- Compute final score ---
    print("Computing final scores...")
    df["final_score"] = df.apply(compute_score, axis=1)

    # --- Assign rank (1 = best) ---
    df["rank"] = df["final_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    # --- Build reasoning ---
    print("Building reasoning strings...")
    df["reasoning"] = df.apply(build_reasoning, axis=1)

    # --- Save scored parquet ---
    df.to_parquet("../artifacts/features_full.parquet", index=False)
    print(f"Saved scored parquet")

    # --- Print summary ---
    print(f"\n--- SCORE DISTRIBUTION ---")
    print(f"  mean  : {df['final_score'].mean():.2f}")
    print(f"  median: {df['final_score'].median():.2f}")
    print(f"  max   : {df['final_score'].max():.2f}")
    print(f"  min   : {df['final_score'].min():.2f}")
    print(f"  > 50  : {(df['final_score'] > 50).sum():,}")
    print(f"  > 60  : {(df['final_score'] > 60).sum():,}")
    print(f"  > 70  : {(df['final_score'] > 70).sum():,}")

    print(f"\n--- TOP 20 CANDIDATES ---")
    top = df.nsmallest(20, "rank")[
        ["rank", "candidate_id", "current_title",
         "years_of_experience", "final_score",
         "must_have_ratio", "semantic_score", "flag_count"]
    ]
    print(top.to_string(index=False))

    return df


if __name__ == "__main__":
    run_scoring()
