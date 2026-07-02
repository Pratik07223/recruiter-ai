"""
Structured spec for the Redrob Senior AI Engineer JD.
Every rule in feature_extraction.py should read from this dict rather
than hardcoding JD text elsewhere, so the JD's reasoning stays traceable.
"""

JD_SPEC = {
    "experience_range_years": (5, 9),  # soft band, not a hard cutoff per JD

    # --- "Things you absolutely need" ---
    "must_have_categories": {
        "embeddings_retrieval": [
            "sentence-transformers", "sentence transformers", "openai embeddings",
            "bge", "e5", "embeddings", "dense retrieval", "vector search",
        ],
        "vector_db_hybrid_search": [
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "hybrid search", "bm25",
        ],
        "python": ["python"],
        "eval_frameworks": [
            "ndcg", "mrr", "map", "a/b test", "ab test", "offline evaluation",
            "online evaluation", "evaluation framework", "learning to rank",
        ],
    },

    # --- "Things we'd like you to have but won't reject you for" ---
    "nice_to_have_categories": {
        "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning"],
        "learning_to_rank": ["xgboost", "learning to rank", "ltr", "neural ranking"],
        "hr_tech": ["hr-tech", "hr tech", "recruiting", "marketplace", "ats"],
        "distributed_systems": ["distributed systems", "large-scale inference", "kubernetes", "spark"],
        "open_source": ["open source", "open-source", "github", "publication", "paper", "conference talk"],
    },

    # --- Hard disqualifiers (JD: "we will not move forward") ---
    "hard_disqualifiers": {
        "pure_research_only": True,
        "langchain_only_recent": True,
        "senior_no_code_18mo": True,
    },

    # --- Soft exclusions (JD: "things we explicitly do NOT want") ---
    "soft_exclusions": {
        "title_chaser": True,
        "framework_enthusiast": True,
        "consulting_only_career": True,
        "cv_speech_robotics_only": True,
        "closed_source_5yr_no_validation": True,
    },

    "consulting_firms": [
        "tcs", "tata consultancy services", "infosys", "wipro",
        "accenture", "cognizant", "capgemini",
    ],

    "core_ai_skill_bucket": [
        "nlp", "natural language processing", "embeddings", "retrieval",
        "ranking", "llm", "large language model", "fine-tuning llms",
        "vector search", "semantic search", "rag",
    ],

    "non_core_ai_skill_bucket": [
        "image classification", "computer vision", "object detection",
        "speech recognition", "tts", "robotics", "gans", "ocr",
    ],

    "location_preference": ["pune", "noida"],
    "location_acceptable": ["hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "bangalore", "bengaluru"],

    "notice_period_ideal_days": 30,
}