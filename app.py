import gradio as gr
import pandas as pd

# Load the submission CSV directly
df = pd.read_csv("submission.csv")

def show_rankings():
    return df

def search_candidate(candidate_id):
    if not candidate_id:
        return df
    result = df[df["candidate_id"].str.contains(candidate_id, case=False)]
    return result

# Build the Gradio interface
with gr.Blocks(title="AI Recruiter — Candidate Ranker") as demo:
    gr.Markdown("""
    # 🤖 AI Recruiter — Senior AI Engineer Candidate Ranker
    **India Runs Data & AI Challenge Submission**
    
    This system ranks 100,000 candidates against a Senior AI Engineer job description
    using a hybrid rule-based + semantic embeddings pipeline.
    
    - ⚡ Processes 100K candidates in under 30 seconds on CPU
    - 🧠 Uses `all-MiniLM-L6-v2` semantic embeddings to catch substance beyond keywords
    - 🚩 Detects honeypot profiles, consulting-only careers, title-chasers
    - ✅ No live LLM API calls at inference
    """)

    gr.Markdown("## 🏆 Top 100 Ranked Candidates")
    
    with gr.Row():
        search_box = gr.Textbox(
            label="Search by Candidate ID (optional)",
            placeholder="e.g. CAND_0052328"
        )
        search_btn = gr.Button("Search", variant="primary")

    output_table = gr.Dataframe(
        value=df,
        headers=["candidate_id", "rank", "score", "reasoning"],
        wrap=True,
        interactive=False,
    )

    search_btn.click(
        fn=search_candidate,
        inputs=search_box,
        outputs=output_table
    )

    gr.Markdown("""
    ## 📐 How It Works
    
    **Step 1 — Rule-Based Feature Extraction**
    Each candidate is scored on evidence strength:
    - `career_backed` = phrase appears in career history (2 pts) 
    - `skill_only` = phrase only in skills list (1 pt)
    - `none` = not found (0 pts)
    
    **Step 2 — Semantic Embeddings**
    Uses `all-MiniLM-L6-v2` locally (no API) to catch candidates
    whose career substance matches the JD without exact keywords.
    
    **Step 3 — Composite Score (0-100)**
    - Must-have evidence: 40 pts
    - Semantic fit: 20 pts  
    - Nice-to-have: 10 pts
    - Production evidence: 5 pts
    - Experience range: 5 pts
    - Location: 5 pts
    - Behavioral signals: 5 pts
    - Flag penalties: -8 pts per flag
    
    **GitHub:** https://github.com/Pratik07223/recruiter-ai
    """)

demo.launch()