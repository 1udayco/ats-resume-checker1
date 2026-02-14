import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")

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
# Extract PDF Text
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

# -----------------------------
# Extract Skills
# -----------------------------
def extract_skills(text):
    return list(set([skill for skill in SKILL_DB if skill in text]))

# -----------------------------
# Extract Experience
# -----------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*years', text)
    if matches:
        return max([int(x) for x in matches])
    return 0

# -----------------------------
# Calculate ATS Score
# -----------------------------
def calculate_ats_score(resume_text, jd_text):

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    semantic_score = similarity * 100

    # Skill Matching
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if len(jd_skills) > 0:
        skill_score = (len(set(resume_skills) & set(jd_skills)) / len(jd_skills)) * 100
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

    final_score = (
        0.4 * semantic_score +
        0.35 * skill_score +
        0.25 * exp_score
    )

    return round(final_score, 2), resume_skills, jd_skills, resume_exp


# -----------------------------
# UI
# -----------------------------
st.title("🚀 AI ATS Resume Analyzer (Cloud Version)")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Resume"):

    if uploaded_file and jd_text.strip() != "":

        resume_text = extract_text_from_pdf(uploaded_file)
        score, resume_skills, jd_skills, resume_exp = calculate_ats_score(
            resume_text, jd_text.lower()
        )

        missing_skills = list(set(jd_skills) - set(resume_skills))

        st.subheader("📊 ATS Score")
        st.progress(int(score))
        st.success(f"Final ATS Score: {score}%")

        st.subheader("🧠 Experience Detected")
        st.info(f"{resume_exp} years")

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
            st.success("Great Match 🎯")

    else:
        st.warning("Upload resume and paste job description.")
