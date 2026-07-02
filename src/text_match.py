"""
text_match.py

Small utilities for searching a candidate's text (skills list + career
history descriptions) for phrases defined in jd_spec.py.

Why this exists: the JD explicitly warns that a candidate's skills[]
list is not reliable evidence on its own. Someone can list "RAG" as a
skill with no real experience, while someone else proves real experience
in their career_history description without ever using that exact word.
So every search in this file looks at BOTH sources.
"""


def get_candidate_text_blob(candidate: dict) -> str:
    """
    Combine every piece of free-text and skill-name text on a candidate
    into one lowercase string we can search through.
    """
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    blob = " ".join(p for p in parts if p)
    return blob.lower()


def contains_any_phrase(text_blob: str, phrases: list) -> bool:
    """Return True if ANY phrase in phrases appears inside text_blob."""
    return any(phrase in text_blob for phrase in phrases)


def count_matching_phrases(text_blob: str, phrases: list) -> int:
    """Return how MANY distinct phrases from phrases appear in text_blob."""
    return sum(1 for phrase in phrases if phrase in text_blob)


def which_phrases_matched(text_blob: str, phrases: list) -> list:
    """Return the actual list of phrases that matched."""
    return [phrase for phrase in phrases if phrase in text_blob]


def get_separated_text_sources(candidate: dict) -> dict:
    """
    Unlike get_candidate_text_blob (which merges everything into one blob),
    this keeps three sources SEPARATE so we can tell evidence apart:

      - skills_text:  just the skills[].name list (cheap to claim, weak evidence)
      - career_text:  career_history[].description + title (what they actually did)
      - profile_text: headline + summary (their own framing of themselves)

    Returns a dict of three lowercase strings.
    """
    profile = candidate.get("profile", {})
    profile_text = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]).lower()

    career_parts = []
    for job in candidate.get("career_history", []):
        career_parts.append(job.get("description", ""))
        career_parts.append(job.get("title", ""))
    career_text = " ".join(p for p in career_parts if p).lower()

    skills_text = " ".join(
        s.get("name", "") for s in candidate.get("skills", [])
    ).lower()

    return {
        "profile_text": profile_text,
        "career_text": career_text,
        "skills_text": skills_text,
    }


def evidence_strength(candidate: dict, phrases: list) -> str:
    """
    Classify how strong the evidence is for a set of phrases:

      "career_backed" — phrase appears in career_history or profile summary
                        (real story behind the claim — strong evidence)
      "skill_only"    — phrase ONLY appears in the skills[] list
                        (cheap to claim — the JD explicitly distrusts this)
      "none"          — phrase doesn't appear anywhere
    """
    sources = get_separated_text_sources(candidate)
    in_career  = contains_any_phrase(sources["career_text"],  phrases)
    in_profile = contains_any_phrase(sources["profile_text"], phrases)
    in_skills  = contains_any_phrase(sources["skills_text"],  phrases)

    if in_career or in_profile:
        return "career_backed"
    elif in_skills:
        return "skill_only"
    else:
        return "none"