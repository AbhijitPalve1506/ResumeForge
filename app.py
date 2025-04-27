# All is Correct
from flask import Flask, request, render_template
import os
import torch
import spacy
import pickle
import numpy as np
import json  # Import json to read the file
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
import PyPDF2
import docx
import requests

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ========== Model Load ==========
print("🔧 Loading models...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

# Load the courses from the JSON file
with open("C:\Career Navigator\data\online_courses_dataset.json", "r") as f:
    course_dict = json.load(f)

# ========== Utility Functions ==========
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def get_roberta_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_text_from_file(file):
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.filename.endswith(".docx"):
        doc_obj = docx.Document(file)
        return "\n".join([para.text for para in doc_obj.paragraphs])
    else:
        return ""

def match_skills_fuzzy(resume_text, required_skills, threshold=80):
    matched = []
    resume_lines = resume_text.lower().splitlines()
    for skill in required_skills:
        for line in resume_lines:
            if fuzz.partial_ratio(skill.lower(), line) >= threshold:
                matched.append(skill)
                break
    return list(set(matched))

# def recommend_courses(missing_skills):
def recommend_courses(missing_skills):
    recommendations = {}
    for skill in missing_skills:
        normalized_skill = skill.lower().replace(" ", "")
        best_match = max((fuzz.partial_ratio(normalized_skill, key) for key in course_dict.keys()), default=0)
        if best_match > 80:  # Threshold for fuzzy match
            matched_key = max((key for key in course_dict.keys() if fuzz.partial_ratio(normalized_skill, key) > 80), key=lambda k: fuzz.partial_ratio(normalized_skill, k), default=skill.lower())
            recommendations[skill] = [course_dict[matched_key]]
        else:
            recommendations[skill] = ["🔍 Course not found. Search manually."]
    return recommendations

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def index():
    return render_template('index.html')



# 1This is corrrect one
@app.route('/upload', methods=['POST'])
def upload():
    resume_file = request.files['resume']
    job_description = request.form['job_description']
    required_skills_input = request.form['required_skills']

    if not resume_file:
        return "❌ No file uploaded", 400

    resume_text = extract_text_from_file(resume_file)
    if not resume_text.strip():
        return "❌ Could not extract text from resume.", 400

    cleaned_resume = preprocess(resume_text)
    cleaned_jd = preprocess(job_description)

    # Semantic Similarity
    resume_embed = get_roberta_embedding(cleaned_resume)
    jd_embed = get_roberta_embedding(cleaned_jd)
    similarity = cosine_similarity([resume_embed], [jd_embed])[0][0]

    # Skill Matching
    required_skills = [skill.strip() for skill in required_skills_input.split(",") if skill.strip()]
    matched_skills = match_skills_fuzzy(resume_text, required_skills, threshold=80)
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]

    # Recommendations
    recommended_courses = recommend_courses(missing_skills)

    # Scores
    skill_match_score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
    semantic_score = round(similarity * 100, 2)
    final_fit_score = round((semantic_score * 0.6) + (skill_match_score * 0.4), 2)

    return render_template(
        'result.html',
        semantic_score=semantic_score,
        skill_match_score=round(skill_match_score, 2),
        final_fit_score=final_fit_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        recommended_courses=recommended_courses
    )

@app.route('/multi-upload', methods=['GET', 'POST'])
def multi_upload():
    if request.method == 'POST':
        job_description = request.form['job_description']
        required_skills_input = request.form['required_skills']
        resume_files = request.files.getlist('resumes')

        if not resume_files or not any(f.filename for f in resume_files):
            return "❌ No files uploaded", 400

        results = []
        cleaned_jd = preprocess(job_description)
        jd_embed = get_roberta_embedding(cleaned_jd)
        required_skills = [skill.strip() for skill in required_skills_input.split(",") if skill.strip()]
        
        for resume_file in resume_files:
            if resume_file and resume_file.filename:
                resume_text = extract_text_from_file(resume_file)
                if not resume_text.strip():
                    continue

                cleaned_resume = preprocess(resume_text)

                # Semantic Similarity
                resume_embed = get_roberta_embedding(cleaned_resume)
                similarity = cosine_similarity([resume_embed], [jd_embed])[0][0]

                # Skill Matching
                matched_skills = match_skills_fuzzy(resume_text, required_skills, threshold=80)
                skill_match_score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
                semantic_score = round(similarity * 100, 2)
                final_fit_score = round((semantic_score * 0.6) + (skill_match_score * 0.4), 2)

                results.append({
                    'filename': resume_file.filename,
                    'final_fit_score': final_fit_score
                })

        # Sort results by final_fit_score in descending order
        results.sort(key=lambda x: x['final_fit_score'], reverse=True)
        # Enumerate results for the template
        enumerated_results = list(enumerate(results, start=1))  # start=1 for 1-based ranking

        return render_template(
            'multi_result.html',
            results=enumerated_results
        )
    return render_template('multi_upload.html')



if __name__ == '__main__':
    app.run(debug=True)