"""
flags.py

Boolean disqualifier checks for each candidate.
Each function returns True if the candidate has that problem.

These are used as PENALTIES in the final score — a flagged candidate
gets pushed down the ranking even if their skill matches look good.
"""

from jd_spec import JD_SPEC


def is_consulting_only(candidate: dict) -> bool:
    """
    True if the candidate's ENTIRE career has been at consulting firms
    (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
    The JD explicitly excludes these candidates.
    """
    firms = JD_SPEC["consulting_firms"]
    history = candidate.get("career_history", [])

    if not history:
        return False

    for job in history:
        company = job.get("company", "").lower()
        # If even ONE job is NOT at a consulting firm, they're not consulting-only
        if not any(firm in company for firm in firms):
            return False

    return True  # every job was at a consulting firm


def is_title_chaser(candidate: dict) -> bool:
    """
    True if the candidate has a pattern of short stints at many companies
    (avg tenure under 1.5 years across 3+ jobs).
    This catches people who hop for title promotions without building depth.
    """
    history = candidate.get("career_history", [])

    if len(history) < 3:
        return False

    durations = [job.get("duration_months", 0) for job in history]
    avg_months = sum(durations) / len(durations)

    return avg_months < 18  # less than 1.5 years average


def is_cv_speech_robotics_only(candidate: dict) -> bool:
    """
    True if the candidate's skills are dominated by CV/speech/robotics
    with NO meaningful NLP/IR/retrieval exposure.
    The JD wants NLP and retrieval — pure vision/speech people are excluded.
    """
    skills_text = " ".join(
        s.get("name", "").lower() for s in candidate.get("skills", [])
    )
    career_text = " ".join(
        job.get("description", "").lower()
        for job in candidate.get("career_history", [])
    )
    all_text = skills_text + " " + career_text

    non_core = JD_SPEC["non_core_ai_skill_bucket"]
    core     = JD_SPEC["core_ai_skill_bucket"]

    has_non_core = any(phrase in all_text for phrase in non_core)
    has_core     = any(phrase in all_text for phrase in core)

    # Flag only if they have CV/speech signals AND no NLP/retrieval signals
    return has_non_core and not has_core


def has_honeypot_inconsistency(candidate: dict) -> bool:
    """
    True if the candidate's profile has impossible or suspicious internal
    inconsistencies — signs of a fake/trap profile.

    Checks:
    1. Claims "expert" proficiency in a skill with 0 duration_months
    2. Total experience years claimed > years since they could have started work
       (graduation year + unrealistic experience window)
    """
    # Check 1: expert skill with 0 months duration
    for skill in candidate.get("skills", []):
        proficiency = skill.get("proficiency", "").lower()
        duration    = skill.get("duration_months", 1)
        if proficiency == "expert" and duration == 0:
            return True

    # Check 2: experience years vs realistic career start
    profile  = candidate.get("profile", {})
    exp_yrs  = profile.get("years_of_experience", 0)
    edu      = candidate.get("education", [])

    if edu:
        # Find the earliest graduation year
        grad_years = [
            e.get("graduation_year", 9999)
            for e in edu
            if e.get("graduation_year")
        ]
        if grad_years:
            earliest_grad = min(grad_years)
            max_possible_exp = 2026 - earliest_grad
            # Flag if claimed experience is more than 2 years beyond possible
            if exp_yrs > max_possible_exp + 2:
                return True

    return False


def is_langchain_only(candidate: dict) -> bool:
    """
    True if the candidate's only AI experience is recent LangChain/OpenAI API
    usage with no underlying ML/retrieval/embeddings engineering.
    The JD calls this out explicitly as a disqualifier.
    """
    all_text = " ".join([
        job.get("description", "").lower()
        for job in candidate.get("career_history", [])
    ])
    skills_text = " ".join(
        s.get("name", "").lower() for s in candidate.get("skills", [])
    )
    combined = all_text + " " + skills_text

    # Signs of LangChain-only AI experience
    langchain_signals = ["langchain", "llamaindex", "openai api", "chatgpt api"]
    # Signs of real underlying ML/retrieval work
    real_ml_signals   = [
        "embeddings", "fine-tuning", "vector", "retrieval",
        "training", "pytorch", "tensorflow", "sentence-transformers",
        "ranking", "ndcg", "mrr",
    ]

    has_langchain = any(sig in combined for sig in langchain_signals)
    has_real_ml   = any(sig in combined for sig in real_ml_signals)

    return has_langchain and not has_real_ml