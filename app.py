from flask import Flask, send_from_directory, request, jsonify
import re
import math
from PyPDF2 import PdfReader
from jobs import JOBS
from openai import OpenAI

# =========================
# APP INIT
# =========================
app = Flask(__name__, static_folder="static")
client = OpenAI()

# =========================
# SKILLS CONFIG
# =========================
SKILLS = [
    "python", "flask", "django", "sql", "postgres", "postgresql", "mysql",
    "mongodb", "javascript", "react", "node", "aws", "docker",
    "kubernetes", "html", "css", "git",
    "machine learning", "deep learning",
    "pandas", "numpy", "tensorflow", "pytorch",
    "rest api", "rest", "api",
    "unit testing", "pytest"
]

SKILL_PATTERNS = {
    skill: re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
    for skill in SKILLS
}

# =========================
# UTILITY FUNCTIONS
# =========================
def split_to_sentences(text):
    parts = re.split(r"[.?!]\s*", text)
    return [p.strip() for p in parts if p.strip()]

def find_skills_with_snippets(text):
    results = {}
    sentences = split_to_sentences(text)
    lower_text = text.lower()

    for skill, pattern in SKILL_PATTERNS.items():
        if not pattern.search(lower_text):
            continue

        snippets = []
        for s in sentences:
            if pattern.search(s):
                snippets.append(s)

        if not snippets:
            snippets = [lower_text[:120]]

        results[skill] = {
            "count": len(snippets),
            "snippets": snippets
        }

    return results

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------- AI SEMANTIC SIMILARITY ----------
def semantic_similarity(text1, text2):
    emb1 = client.embeddings.create(
        model="text-embedding-3-small",
        input=text1
    ).data[0].embedding

    emb2 = client.embeddings.create(
        model="text-embedding-3-small",
        input=text2
    ).data[0].embedding

    dot = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = math.sqrt(sum(a * a for a in emb1))
    norm2 = math.sqrt(sum(b * b for b in emb2))

    return dot / (norm1 * norm2)

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# -------------------------
# RESUME vs JOB ANALYSIS
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    resume_text = data.get("resume", "")
    job_text = data.get("job", "")

    resume_info = find_skills_with_snippets(resume_text)
    job_info = find_skills_with_snippets(job_text)

    resume_skills = list(resume_info.keys())
    job_skills = list(job_info.keys())

    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]

    rule_score = int((len(matched) / len(job_skills)) * 100) if job_skills else 0
    ai_score = int(semantic_similarity(resume_text, job_text) * 100)

    matched_details = {s: resume_info[s] for s in matched}

    return jsonify({
        "rule_based_score": rule_score,
        "ai_semantic_score": ai_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_details": matched_details
    })

# -------------------------
# ANALYZE WITH PDF
# -------------------------
@app.route("/analyze-file", methods=["POST"])
def analyze_file():
    job_text = request.form.get("job", "")

    if "resume_file" in request.files and request.files["resume_file"].filename:
        resume_text = extract_text_from_pdf(request.files["resume_file"])
    else:
        resume_text = request.form.get("resume_text", "")

    resume_info = find_skills_with_snippets(resume_text)
    job_info = find_skills_with_snippets(job_text)

    resume_skills = list(resume_info.keys())
    job_skills = list(job_info.keys())

    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]

    rule_score = int((len(matched) / len(job_skills)) * 100) if job_skills else 0
    ai_score = int(semantic_similarity(resume_text, job_text) * 100)

    matched_details = {s: resume_info[s] for s in matched}

    return jsonify({
        "rule_based_score": rule_score,
        "ai_semantic_score": ai_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_details": matched_details
    })

# -------------------------
# JOB RECOMMENDATION (PDF)
# -------------------------
@app.route("/recommend-jobs-file", methods=["POST"])
def recommend_jobs_file():
    if "resume_file" in request.files and request.files["resume_file"].filename:
        resume_text = extract_text_from_pdf(request.files["resume_file"])
    else:
        resume_text = request.form.get("resume_text", "")

    resume_info = find_skills_with_snippets(resume_text)
    resume_skills = list(resume_info.keys())

    recommendations = []

    for job in JOBS:
        must = job["must_have"]
        bonus = job["good_to_have"]

        matched_must = [s for s in must if s in resume_skills]
        matched_bonus = [s for s in bonus if s in resume_skills]
        missing_must = [s for s in must if s not in resume_skills]

        must_score = (len(matched_must) / len(must)) * 70 if must else 70
        bonus_score = (len(matched_bonus) / len(bonus)) * 30 if bonus else 0
        score = int(must_score + bonus_score)

        recommendations.append({
            "job_title": job["title"],
            "score": score,
            "matched_must_have": matched_must,
            "matched_good_to_have": matched_bonus,
            "missing_must_have": missing_must
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(recommendations[:3])

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
