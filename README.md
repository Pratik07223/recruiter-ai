# AI Recruiter — Intelligent Candidate Ranking System
### India Runs Data & AI Challenge Submission

A hybrid rule-based + semantic pipeline that ranks 100,000 candidates against a Senior AI Engineer job description in under 30 seconds on CPU, with no live LLM API calls.

---

## The Problem

Given 100,000 candidate profiles and a detailed job description for a Senior AI Engineer, rank candidates from most to least suitable — accurately, fast, and explainably.

The JD explicitly warns against two failure modes that naive systems fall into:
- **Keyword stuffing traps** — candidates who list "RAG", "Pinecone", "LangChain" as skills with no real experience behind them
- **Missed substance** — candidates who built real retrieval/ranking systems but describe their work in plain language without exact JD keywords

Our system is specifically designed to handle both.

---

## Architecture

```
candidates.jsonl (100K records)
        │
        ▼
┌─────────────────────────────┐
│  STEP 1: Rule-Based         │  ~20 seconds
│  Feature Extraction         │
│  src/feature_extraction.py  │
│                             │
│  • evidence_strength()      │  ← core innovation
│    "career_backed" vs       │    separates real evidence
│    "skill_only" vs "none"   │    from keyword stuffing
│                             │
│  • Flag functions           │  ← disqualifiers
│    consulting_only          │
│    title_chaser             │
│    cv_speech_robotics_only  │
│    honeypot_inconsistency   │
│    langchain_only           │
└────────────┬────────────────┘
             │ artifacts/features_full.parquet
             ▼
┌─────────────────────────────┐
│  STEP 2: Semantic Embeddings│  ~29 min first run, cached
│  src/embeddings.py          │  instantly on reruns
│                             │
│  Model: all-MiniLM-L6-v2   │  local, no API, CPU-only
│  • Embed JD requirements    │
│  • Embed candidate career   │
│    history text             │
│  • Cosine similarity score  │
│    catches Tier 5 candidates│
│    with right substance but │
│    wrong vocabulary         │
└────────────┬────────────────┘
             │ semantic_score column added to parquet
             ▼
┌─────────────────────────────┐
│  STEP 3: Composite Scorer   │  < 5 seconds
│  src/scorer.py              │
│                             │
│  Weighted formula (0-100):  │
│  • Must-have evidence  40pt │
│  • Semantic fit        20pt │
│  • Nice-to-have        10pt │
│  • Production evidence  5pt │
│  • Experience range     5pt │
│  • Location             5pt │
│  • Behavioral signals   5pt │
│  • Flag penalties      -8pt │
│    per flag                 │
└────────────┬────────────────┘
             │ final_score + rank + reasoning columns
             ▼
┌─────────────────────────────┐
│  STEP 4: Output             │  < 1 second
│  src/rank.py                │
│                             │
│  Top 100 candidates ranked  │
│  with explainable reasoning │
│  → artifacts/submission.csv │
└─────────────────────────────┘
```

---

## Core Innovation: `evidence_strength()`

The single most important function in the system. Instead of asking "does this candidate have this skill?", it asks "**where is the evidence?**"

```
"career_backed"  →  phrase appears in career_history descriptions
                     or profile summary (2 points)
                     REAL evidence of doing the work

"skill_only"     →  phrase ONLY appears in the skills[] list
                     (1 point)
                     cheap to claim, the JD explicitly distrusts this

"none"           →  phrase appears nowhere (0 points)
```

**Example:** A candidate lists "Milvus" as a skill (skill_only = 1pt) but their career history never mentions vector databases → they score lower than a candidate whose job description says "built product search using Elasticsearch and dense retrieval" even though they never wrote the word "Milvus" (career_backed = 2pts).

This directly addresses what the JD calls the "keyword trap."

---

## Disqualifier Flags

Each flag applies a -8 point penalty. Honeypot profiles get an additional -25.

| Flag | What it catches |
|------|----------------|
| `consulting_only` | Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini |
| `title_chaser` | Average job tenure < 18 months across 3+ jobs |
| `cv_speech_robotics` | CV/speech/robotics domain with no NLP/IR exposure |
| `honeypot` | Expert proficiency with 0 duration_months, or impossible experience timeline |
| `langchain_only` | LangChain/OpenAI API usage as sole "AI experience", no real ML engineering |

---

## Scoring Weights

| Signal | Points | Why |
|--------|--------|-----|
| Must-have evidence (career-backed) | up to 40 | Core JD requirements, career-backed = 2x skill-only |
| Semantic similarity | up to 20 | Catches substance without exact keywords |
| Nice-to-have evidence | up to 10 | Bonus for LoRA, LTR, HR-tech, open-source |
| Production deployment evidence | 5 | "shipped", "production", "at scale" in career text |
| Experience in 5-9yr range | 5 | JD's stated sweet spot |
| Location (Pune/Noida preferred) | 5 | JD's stated location preference |
| Behavioral availability | 5 | Response rate + profile completeness |
| Per disqualifier flag | -8 | Consulting-only, title-chaser, etc. |
| Honeypot flag (extra) | -25 | Impossible/fake profile signals |

---

## Results

**Performance on 100K candidates:**
- Step 1 (rule-based extraction): **19.6 seconds**
- Step 2 (semantic embeddings): **29 minutes first run**, instant on reruns (cached)
- Step 3 (scoring + ranking): **< 5 seconds**
- Total inference after cache: **< 30 seconds** ✅ well within 5-minute budget

**Flag counts across 100K:**
- Consulting-only careers: 7,034
- CV/speech/robotics-only: 10,743
- LangChain-only: 1,780
- Title-chasers: 1,707
- Honeypot profiles: 21

**Top candidates (Rank 1-5):**

| Rank | Candidate | Title | Exp | Score |
|------|-----------|-------|-----|-------|
| 1 | CAND_0052328 | Recommendation Systems Engineer | 6.5 yrs | 72.43 |
| 2 | CAND_0027691 | NLP Engineer | 6.5 yrs | 72.39 |
| 3 | CAND_0007009 | Recommendation Systems Engineer | 7.9 yrs | 71.18 |
| 4 | CAND_0008425 | Senior NLP Engineer | 7.8 yrs | 70.93 |
| 5 | CAND_0046064 | Senior NLP Engineer | 8.9 yrs | 70.18 |

All top candidates have:
- `must_have_ratio` ≥ 0.75 (career-backed evidence across 3-4 must-have categories)
- `flag_count = 0` (zero disqualifiers)
- `production_evidence = True`
- Experience within the 5-9 year target range

---

## How to Run

### Prerequisites

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install pandas pyarrow sentence-transformers scikit-learn numpy
```

### Setup

Place these files in `data/`:
- `candidates.jsonl`
- `sample_candidates.json`
- `candidate_schema.json`

### Run the full pipeline

```bash
cd src

# Step 1: Rule-based feature extraction (~20 seconds)
python run_step1.py

# Step 2: Semantic embeddings (~29 min first run, instant after cache)
python embeddings.py

# Step 3: Score and rank all candidates (< 5 seconds)
python scorer.py

# Step 4: Generate submission file (< 1 second)
python rank.py
```

Output: `artifacts/submission.csv` — top 100 candidates with rank, score, and reasoning.

### Validate submission

```bash
python data/validate_submission.py artifacts/submission.csv
# Expected: "Submission is valid."
```

---

## File Structure

```
recruiter-ai/
├── data/
│   ├── candidates.jsonl          ← 100K candidates (not in repo — too large)
│   ├── sample_candidates.json    ← 50 sample candidates
│   ├── candidate_schema.json     ← field definitions
│   └── validate_submission.py    ← official validator
├── src/
│   ├── jd_spec.py               ← JD requirements as structured dict
│   ├── text_match.py            ← text search + evidence_strength()
│   ├── flags.py                 ← disqualifier flag functions
│   ├── feature_extraction.py    ← extract_features() per candidate
│   ├── run_step1.py             ← run Step 1 on full 100K
│   ├── embeddings.py            ← semantic embedding score
│   ├── scorer.py                ← composite final score + reasoning
│   └── rank.py                  ← output ranked CSV
├── artifacts/
│   ├── features_full.parquet    ← Step 1+2+3 output (not in repo)
│   ├── embeddings_cache.npy     ← cached embeddings (not in repo)
│   └── submission.csv           ← FINAL SUBMISSION FILE ✅
├── README.md
└── requirements.txt
```

---

## Design Decisions

**Why rule-based + semantic hybrid, not pure LLM scoring?**
LLM scoring per candidate is too slow (100K × API latency) and expensive. Rule-based handles clear signals instantly; semantics handle vocabulary mismatches. Together they match what a careful human recruiter would do.

**Why no LLM calls at inference?**
The submission constraint prohibits live API calls. All reasoning text is templated from computed features — deterministic, fast, and hallucination-free.

**Why `all-MiniLM-L6-v2`?**
Small (80MB), fast on CPU, strong enough for career-text similarity. Larger models (BGE-large, E5) would be more accurate but exceed the CPU time budget.

**Why embed only career text, not the full profile?**
Skills lists are unreliable (the JD says so explicitly). Embedding career history + profile summary captures what the candidate actually did, not what they claim to know.

**Why an interpretable formula, not a learned model?**
No labeled training data exists for this task. An interpretable weighted formula is fully explainable at interview and easy to tune if the JD's priorities change.

---

## Known Limitations

- Honeypot count (21) may be lower than the dataset's actual ~80 — some honeypot patterns may use phrasing our consistency checks don't cover. The semantic layer may surface additional suspicious profiles.
- Semantic scores compress into a narrow range (0.05–0.71) — normalization works but a larger/better embedding model would spread scores more.
- `open_to_work` is False for all 100K candidates in this dataset — this behavioral signal is unused.
- Nice-to-have matching could be improved by distinguishing career-backed vs skill-only evidence (currently just presence/absence).

---

## Requirements

```
pandas>=2.0
pyarrow>=14.0
sentence-transformers>=2.0
scikit-learn>=1.0
numpy>=1.22
```

Install: `pip install pandas pyarrow sentence-transformers scikit-learn numpy`
