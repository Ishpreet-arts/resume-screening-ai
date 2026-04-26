"""
===========================================
  STEP 5: DEPLOYMENT — STREAMLIT APP
  Resume Screening AI Project
===========================================
How to run:
    streamlit run app.py

What this app does:
  - User pastes resume text + job requirements
  - App predicts match score AND match label
  - Shows skill overlap analysis
  - Gives a recommendation
"""

import streamlit as st
import pickle
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)

# ─── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "../models")
    with open(f"{base}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(f"{base}/regression_model.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(f"{base}/classifier_model.pkl", "rb") as f:
        cls = pickle.load(f)
    return tfidf, reg, cls

tfidf, reg_model, cls_model = load_models()

# ─── Text Cleaner ─────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"[\[\]'\"{}()]", " ", str(text))
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def get_skill_overlap(resume_text, job_text):
    """Find common skills between resume and job description."""
    common_skills = [
        "python", "java", "sql", "machine learning", "deep learning",
        "tensorflow", "keras", "pytorch", "nlp", "data science",
        "excel", "power bi", "tableau", "r", "scala", "spark",
        "aws", "azure", "docker", "kubernetes", "git", "agile",
        "communication", "leadership", "management", "analysis",
        "javascript", "react", "html", "css", "node",
        "finance", "accounting", "marketing", "sales", "hr"
    ]
    resume_lower = resume_text.lower()
    job_lower    = job_text.lower()
    overlap = [s for s in common_skills if s in resume_lower and s in job_lower]
    only_resume = [s for s in common_skills if s in resume_lower and s not in job_lower]
    only_job    = [s for s in common_skills if s not in resume_lower and s in job_lower]
    return overlap, only_job

# ─── Header ───────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#1f3a5f;'>📄 Resume Screening AI</h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Instantly predict how well a resume matches a job requirement
    </p>
    <hr style='margin-bottom:30px;'>
""", unsafe_allow_html=True)

# ─── Two Column Input ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Paste Resume Text")
    resume_input = st.text_area(
        label="Resume",
        height=300,
        placeholder="Paste the candidate's resume here...\n\nInclude: skills, experience, education, career objective...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("💼 Paste Job Requirements")
    job_input = st.text_area(
        label="Job Requirements",
        height=300,
        placeholder="Paste the job description here...\n\nInclude: required skills, experience needed, responsibilities...",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Predict Button ───────────────────────────────────────
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    predict_btn = st.button("🔍  Analyze Match", use_container_width=True, type="primary")

# ─── Results ──────────────────────────────────────────────
if predict_btn:
    if not resume_input.strip() or not job_input.strip():
        st.warning("⚠️ Please fill in both the resume and job requirements!")
    else:
        # Clean and combine
        clean_resume = clean_text(resume_input)
        clean_job    = clean_text(job_input)
        combined     = clean_resume + " " + clean_job

        # Vectorize and predict
        X = tfidf.transform([combined])
        score = reg_model.predict(X)[0]
        label = cls_model.predict(X)[0]
        score = float(np.clip(score, 0, 1))

        # Skill analysis
        overlap, missing = get_skill_overlap(resume_input, job_input)

        st.markdown("---")
        st.subheader("📊 Match Analysis Results")

        # Score + Label
        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("🎯 Match Score", f"{score:.0%}")

        with r2:
            color_map = {"High Match": "🟢", "Medium Match": "🟡", "Low Match": "🔴"}
            icon = color_map.get(label, "⚪")
            st.metric("🏷️ Match Category", f"{icon} {label}")

        with r3:
            st.metric("🔗 Skill Matches Found", str(len(overlap)))

        # Score bar
        st.markdown("<br>", unsafe_allow_html=True)
        bar_color = "#27AE60" if score >= 0.75 else ("#F39C12" if score >= 0.50 else "#E74C3C")
        st.markdown(f"""
        <div style="background:#eee; border-radius:10px; height:24px; width:100%;">
            <div style="background:{bar_color}; width:{score*100:.0f}%; height:100%; border-radius:10px;
                        display:flex; align-items:center; padding-left:10px; color:white; font-weight:bold;">
                {score:.0%} Match
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation box
        st.markdown("<br>", unsafe_allow_html=True)
        if score >= 0.75:
            st.success("✅ **Strong Match!** This candidate is well-suited for the role. Recommended for interview.")
        elif score >= 0.50:
            st.warning("🟡 **Moderate Match.** The candidate meets some requirements. Consider a screening call.")
        else:
            st.error("❌ **Low Match.** The candidate's profile doesn't closely align with this role.")

        # Skill breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### ✅ Matching Skills Found")
            if overlap:
                for skill in overlap:
                    st.markdown(f"- `{skill}`")
            else:
                st.info("No common skills detected from the predefined list.")

        with c2:
            st.markdown("#### ⚠️ Required Skills Not Found in Resume")
            if missing:
                for skill in missing[:10]:
                    st.markdown(f"- `{skill}`")
            else:
                st.success("All required skills appear to be present!")

# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:gray; font-size:13px;'>
    Built with ❤️ using Python, Scikit-learn & Streamlit &nbsp;|&nbsp; Resume Screening AI Project
</p>
""", unsafe_allow_html=True)
