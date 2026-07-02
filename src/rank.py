"""
rank.py

Step 4: Generate the final submission CSV in the exact required format.

Reads the scored parquet, takes top 1000 candidates, outputs the
submission file. Run validate_submission.py after this to confirm
the format is correct before submitting.
"""

import pandas as pd
import os


def generate_submission(
    top_n: int = 100,
    input_path: str = "../artifacts/features_full.parquet",
    output_path: str = "../artifacts/submission.csv",
):
    print("Loading scored features...")
    df = pd.read_parquet(input_path)

    # Confirm scoring columns exist
    required = ["candidate_id", "rank", "final_score", "reasoning"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: missing column '{col}'. Run scorer.py first.")
            return

    print(f"Loaded {len(df):,} candidates")
    print(f"Selecting top {top_n} by rank...")

    # Take top N candidates
    top_df = df.nsmallest(top_n, "rank").copy()

    # Re-rank cleanly from 1 to top_n
    top_df = top_df.sort_values("rank").reset_index(drop=True)
    top_df["rank"] = top_df.index + 1

    # Round final score to 4 decimal places
    top_df["final_score"] = top_df["final_score"].round(4)

    # Build submission dataframe with exact required columns
    submission = pd.DataFrame({
        "candidate_id": top_df["candidate_id"],
        "rank":         top_df["rank"],
        "score":        top_df["final_score"],
        "reasoning":    top_df["reasoning"],
    })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")
    print(f"Shape: {submission.shape}")

    # Preview
    print(f"\n--- TOP 10 IN SUBMISSION ---")
    print(submission.head(10).to_string(index=False))

    print(f"\n--- SUBMISSION COLUMN NAMES ---")
    print(list(submission.columns))

    print(f"\n--- SAMPLE REASONING (rank 1) ---")
    print(submission.iloc[0]["reasoning"])

    return submission


if __name__ == "__main__":
    generate_submission()