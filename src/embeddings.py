"""
embeddings.py

Step 2: Semantic similarity between JD and each candidate's career text.

Why: catches candidates who have the right experience but don't use
the exact keywords in jd_spec.py (the JD calls these "Tier 5" candidates).

Model: all-MiniLM-L6-v2
  - Small (80MB), fast on CPU, good enough for this task
  - No API calls, runs fully local
  - Produces 384-dimension vectors

Output: adds a 'semantic_score' column to features_full.parquet
"""

import json
import numpy as np
import pandas as pd
import os
import time
from sentence_transformers import SentenceTransformer
from text_match import get_separated_text_sources
from jd_spec import JD_SPEC


# -----------------------------------------------------------------------
# 1. Build the JD text we will embed
#    We use the must-have and nice-to-have phrases as a structured
#    description of what we're looking for — not the raw JD doc text,
#    which is too long and noisy for embedding comparison.
# -----------------------------------------------------------------------
def build_jd_text() -> str:
    parts = []

    parts.append("Senior AI Engineer with expertise in:")

    for category, phrases in JD_SPEC["must_have_categories"].items():
        parts.append(f"{category}: {', '.join(phrases[:5])}")

    for category, phrases in JD_SPEC["nice_to_have_categories"].items():
        parts.append(f"{category}: {', '.join(phrases[:3])}")

    parts.append("production deployment of machine learning systems at scale")
    parts.append("building and evaluating retrieval and ranking systems")
    parts.append("shipped real products to real users")

    return " | ".join(parts)


# -----------------------------------------------------------------------
# 2. Build candidate text for embedding
#    We embed only career_text + profile_text (NOT skills_text)
#    because skills are unreliable — we already handle them in Step 1.
#    We want the embedding to capture career substance only.
# -----------------------------------------------------------------------
def build_candidate_text(candidate: dict) -> str:
    sources = get_separated_text_sources(candidate)
    # Combine career history + profile summary
    # Truncate to 512 words to stay within model token limit
    combined = sources["career_text"] + " " + sources["profile_text"]
    words = combined.split()[:512]
    return " ".join(words)


# -----------------------------------------------------------------------
# 3. Main embedding function
#    Loads candidates, builds texts, embeds in batches, saves scores
# -----------------------------------------------------------------------
def run_embeddings():
    print("Loading model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    # --- Embed the JD once ---
    jd_text = build_jd_text()
    print(f"\nJD text for embedding:\n{jd_text[:200]}...\n")
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)[0]
    print(f"JD embedding shape: {jd_embedding.shape}")

    # --- Load candidates ---
    print("\nLoading candidates...")
    candidates = []
    with open("../data/candidates.jsonl") as f:
        for line in f:
            candidates.append(json.loads(line.strip()))
    print(f"Loaded {len(candidates):,} candidates")

    # --- Check if cache exists (skip recomputing if already done) ---
    cache_path = "../artifacts/embeddings_cache.npy"
    ids_path   = "../artifacts/embeddings_ids.npy"

    if os.path.exists(cache_path) and os.path.exists(ids_path):
        print("\nCache found — loading embeddings from disk...")
        candidate_embeddings = np.load(cache_path)
        candidate_ids = np.load(ids_path, allow_pickle=True)
        print(f"Loaded cache: {candidate_embeddings.shape}")
    else:
        # --- Build candidate texts ---
        print("\nBuilding candidate texts...")
        candidate_texts = []
        candidate_ids   = []
        for c in candidates:
            candidate_texts.append(build_candidate_text(c))
            candidate_ids.append(c.get("candidate_id", "UNKNOWN"))

        # --- Embed in batches ---
        print(f"Embedding {len(candidate_texts):,} candidates in batches...")
        start = time.time()

        candidate_embeddings = model.encode(
            candidate_texts,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        elapsed = time.time() - start
        print(f"Embedding done in {elapsed:.1f} seconds")
        print(f"Embeddings shape: {candidate_embeddings.shape}")

        # --- Save cache so we never recompute ---
        np.save(cache_path, candidate_embeddings)
        np.save(ids_path, np.array(candidate_ids))
        print(f"Cache saved to {cache_path}")

    # --- Compute cosine similarity ---
    # Both JD and candidates are already normalized (normalize_embeddings=True)
    # so cosine similarity = dot product
    print("\nComputing similarity scores...")
    semantic_scores = candidate_embeddings @ jd_embedding
    print(f"Score range: min={semantic_scores.min():.3f}  max={semantic_scores.max():.3f}  mean={semantic_scores.mean():.3f}")

    # --- Load existing features parquet and add semantic_score column ---
    print("\nLoading features_full.parquet...")
    df = pd.read_parquet("../artifacts/features_full.parquet")

    # Match by candidate_id order (should be same order as candidates.jsonl)
    df["semantic_score"] = semantic_scores
    df["semantic_score"] = df["semantic_score"].round(4)

    # Save updated parquet
    df.to_parquet("../artifacts/features_full.parquet", index=False)
    print(f"Saved updated parquet with semantic_score column")
    print(f"New shape: {df.shape}")

    # --- Preview top candidates by semantic score ---
    print("\n--- TOP 10 BY SEMANTIC SCORE ---")
    top = df.nlargest(10, "semantic_score")[
        ["candidate_id", "current_title", "years_of_experience",
         "semantic_score", "must_have_ratio", "flag_count"]
    ]
    print(top.to_string(index=False))

    print("\n--- TOP 10 BY MUST-HAVE RATIO (for comparison) ---")
    top2 = df.nlargest(10, "must_have_ratio")[
        ["candidate_id", "current_title", "years_of_experience",
         "semantic_score", "must_have_ratio", "flag_count"]
    ]
    print(top2.to_string(index=False))


if __name__ == "__main__":
    run_embeddings()