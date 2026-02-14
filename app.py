import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    layout="wide"
)

# -----------------------------
# Load AI Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Skill Database
# -----------------------------
SKILL_DB = [
    "python","sql","machine learning","deep learning",
    "tensorflow","pytorch","react","springboot",
    "aws","docker","kubernetes","flask",
    "data analysis","nlp","pandas","numpy",
    "power bi","excel","c++","java"
]

# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.lower()

# -----------------------------
# Extract Skills
# -----------------------------
def extract_skills(text):
    found = []
    for skill in SKILL_DB:
        if skill in text:
            found.append(skill)
    return list(set(found))

# -----------------------------
# Extract Experience
# -----------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*years', text)
    if matches:
        return max([int(x) for x in matches])
    return 0

# -----------------------------
# ATS Score Calculation
# -----------------------------
def calculate_ats_score(resume_text, jd_text):

    # Semantic Similarity
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    semantic_score = similarity * 100

    # Skill Matching
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if len(jd_skills) > 0:
        matched = len(set(resume_skills) & set(jd_skills))
        skill_score = (matched / len(jd_skills)) * 100
    else:
        skill_score = 50

    # Experience Matching
    resume_exp = extract_experience(resume_text)
    jd_exp = extract_experience(jd_text)

    if jd_exp == 0:
        exp_score = 100
    elif resume_exp >= jd_exp:
        exp_score = 100
    else:
        exp_score = (resume_exp / jd_exp) * 100

    # Final Score (Weighted)
    final_score = (
        0.35 * semantic_score +
        0.30 * skill_score +
        0.20 * exp_score +
        0.15 * 100
    )

    return round(final_score, 2), resume_skills, jd_skills, resume_exp


# -----------------------------
# UI Layout
# -----------------------------
st.title("🚀 AI ATS Resume Analyzer")
st.write("Upload your Resume (PDF) and paste Job Description below.")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
jd_text = st.text_area("Paste Job Description Here", height=200)

if st.button("Analyze Resume"):

    if uploaded_file is not None and jd_text.strip() != "":

        resume_text = extract_text_from_pdf(uploaded_file)

        score, resume_skills, jd_skills, resume_exp = calculate_ats_score(
            resume_text, jd_text.lower()
        )

        missing_skills = list(set(jd_skills) - set(resume_skills))

        st.subheader("📊 ATS Score")
        st.progress(int(score))
        st.success(f"Final ATS Score: {score}%")

        st.subheader("🧠 Detected Experience")
        st.info(f"{resume_exp} years detected")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Resume Skills")
            st.write(resume_skills)

        with col2:
            st.subheader("📋 JD Skills")
            st.write(jd_skills)

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.error(missing_skills)
        else:
            st.success("No Missing Skills! Great Match 🎯")

    else:
        st.warning("Please upload resume and paste job description.")
