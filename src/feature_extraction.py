"""
feature_extraction.py

The main Step 1 function: takes one candidate dict, returns a flat
dictionary of all features (skill matches + flags + experience numbers).

This output is what the scorer (Step 3) will consume.
Every feature here is rule-based — fast, deterministic, explainable.
No ML, no API calls.
"""

from jd_spec import JD_SPEC
from text_match import evidence_strength, which_phrases_matched, get_candidate_text_blob
from flags import (
    is_consulting_only,
    is_title_chaser,
    is_cv_speech_robotics_only,
    has_honeypot_inconsistency,
    is_langchain_only,
)


def extract_features(candidate: dict) -> dict:
    """
    Takes one raw candidate dict from the JSONL file.
    Returns a flat dict of features ready for scoring.
    """
    profile  = candidate.get("profile", {})
    cand_id  = candidate.get("candidate_id", "UNKNOWN")

    # ------------------------------------------------------------------
    # 1. BASIC PROFILE FIELDS
    # ------------------------------------------------------------------
    exp_years     = profile.get("years_of_experience", 0) or 0
    location      = profile.get("location", "").lower()
    current_title = profile.get("current_title", "").lower()
    industry      = profile.get("current_industry", "").lower()

    # ------------------------------------------------------------------
    # 2. EXPERIENCE RANGE FIT
    # min=5, max=9 per JD — outside this range is a soft penalty
    # ------------------------------------------------------------------
    exp_min, exp_max = JD_SPEC["experience_range_years"]
    exp_in_range = exp_min <= exp_years <= exp_max

    # ------------------------------------------------------------------
    # 3. LOCATION SIGNALS
    # ------------------------------------------------------------------
    preferred_location = any(
        loc in location for loc in JD_SPEC["location_preference"]
    )
    acceptable_location = any(
        loc in location for loc in JD_SPEC["location_acceptable"]
    )

    # ------------------------------------------------------------------
    # 4. MUST-HAVE CATEGORY EVIDENCE STRENGTH
    # For each must-have category, score:
    #   career_backed = 2 points (real evidence)
    #   skill_only    = 1 point  (claimed but unverified)
    #   none          = 0 points
    # ------------------------------------------------------------------
    must_have_scores = {}
    must_have_raw    = {}
    for category, phrases in JD_SPEC["must_have_categories"].items():
        strength = evidence_strength(candidate, phrases)
        must_have_raw[category] = strength
        if strength == "career_backed":
            must_have_scores[category] = 2
        elif strength == "skill_only":
            must_have_scores[category] = 1
        else:
            must_have_scores[category] = 0

    total_must_have_score = sum(must_have_scores.values())
    # Max possible = 2 points x 4 categories = 8
    must_have_ratio = total_must_have_score / 8

    # ------------------------------------------------------------------
    # 5. NICE-TO-HAVE CATEGORY EVIDENCE STRENGTH
    # Same logic, just less weight in the final score
    # ------------------------------------------------------------------
    nice_scores = {}
    for category, phrases in JD_SPEC["nice_to_have_categories"].items():
        strength = evidence_strength(candidate, phrases)
        if strength == "career_backed":
            nice_scores[category] = 2
        elif strength == "skill_only":
            nice_scores[category] = 1
        else:
            nice_scores[category] = 0

    total_nice_score  = sum(nice_scores.values())
    # Max possible = 2 x 5 categories = 10
    nice_ratio = total_nice_score / 10

    # ------------------------------------------------------------------
    # 6. CAREER HISTORY DEPTH SIGNALS
    # ------------------------------------------------------------------
    history  = candidate.get("career_history", [])
    num_jobs = len(history)
    durations = [job.get("duration_months", 0) for job in history]
    avg_tenure_months = sum(durations) / len(durations) if durations else 0

    # Production evidence: phrases in career history that suggest
    # real deployed systems, not just research or tutorials
    production_phrases = [
        "production", "deployed", "shipped", "at scale",
        "million users", "real-time", "serving", "inference",
        "launched", "built and deployed",
    ]
    blob = get_candidate_text_blob(candidate)
    production_evidence = any(p in blob for p in production_phrases)

    # ------------------------------------------------------------------
    # 7. DISQUALIFIER FLAGS
    # ------------------------------------------------------------------
    flag_consulting_only    = is_consulting_only(candidate)
    flag_title_chaser       = is_title_chaser(candidate)
    flag_cv_speech_robotics = is_cv_speech_robotics_only(candidate)
    flag_honeypot           = has_honeypot_inconsistency(candidate)
    flag_langchain_only     = is_langchain_only(candidate)

    # Count how many flags are active — more flags = bigger penalty later
    flag_count = sum([
        flag_consulting_only,
        flag_title_chaser,
        flag_cv_speech_robotics,
        flag_honeypot,
        flag_langchain_only,
    ])

    # ------------------------------------------------------------------
    # 8. BEHAVIORAL / AVAILABILITY SIGNALS
    # from redrob_signals fields on the candidate
    # ------------------------------------------------------------------
    signals = candidate.get("redrob_signals", {})
    open_to_work         = signals.get("open_to_work", False)
    actively_applying    = signals.get("actively_applying", False)
    response_rate        = signals.get("recruiter_response_rate", 0) or 0
    profile_completeness = signals.get("profile_completeness_score", 0) or 0

    # ------------------------------------------------------------------
    # 9. ASSEMBLE FINAL FEATURE ROW
    # ------------------------------------------------------------------
    return {
        "candidate_id": cand_id,
        "current_title": profile.get("current_title", ""),
        "years_of_experience": exp_years,
        "location": profile.get("location", ""),
        "industry": industry,

        # Experience fit
        "exp_in_range": exp_in_range,

        # Location
        "preferred_location": preferred_location,
        "acceptable_location": acceptable_location,

        # Must-have scores (0/1/2 per category)
        "score_embeddings_retrieval": must_have_scores.get("embeddings_retrieval", 0),
        "score_vector_db":            must_have_scores.get("vector_db_hybrid_search", 0),
        "score_python":               must_have_scores.get("python", 0),
        "score_eval_frameworks":      must_have_scores.get("eval_frameworks", 0),
        "total_must_have_score":      total_must_have_score,
        "must_have_ratio":            round(must_have_ratio, 3),

        # Nice-to-have scores
        "total_nice_score": total_nice_score,
        "nice_ratio":       round(nice_ratio, 3),

        # Career depth
        "num_jobs":           num_jobs,
        "avg_tenure_months":  round(avg_tenure_months, 1),
        "production_evidence": production_evidence,

        # Flags (True = bad signal)
        "flag_consulting_only":    flag_consulting_only,
        "flag_title_chaser":       flag_title_chaser,
        "flag_cv_speech_robotics": flag_cv_speech_robotics,
        "flag_honeypot":           flag_honeypot,
        "flag_langchain_only":     flag_langchain_only,
        "flag_count":              flag_count,

        # Behavioral signals
        "open_to_work":          open_to_work,
        "actively_applying":     actively_applying,
        "response_rate":         response_rate,
        "profile_completeness":  profile_completeness,
    }