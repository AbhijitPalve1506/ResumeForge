# ResumeForge

🔍 Project Overview

**Smart Resume Analyzer** is a Flask-based web application designed for:
- **Candidates**: Upload their resume, provide a job description & required skills → get a **semantic match score**, **skill match score**, **final fit score**, **matched & missing skills**, and **recommended online courses** to improve missing skills.
- **HR Professionals**: Upload **multiple resumes** for a specific job role → get resumes **ranked** based on their final fit score to assist in smart hiring.

🚀 Features

- **Semantic Matching**: Calculates how closely a resume matches the job description using BERT-based embeddings (RoBERTa).
- **Skill Matching**: Extracts matched and missing skills from resumes based on fuzzy matching.
- **Course Recommendation**: Recommends online courses for missing skills from a curated dataset.
- **Resume Ranking**: Ranks multiple candidates based on both semantic similarity and skill matching.
- **PDF and DOCX Support**: Extracts text from uploaded resumes.
- **User Friendly**: Simple web-based upload system for candidates and HRs.

🛠️ Tools & Technologies Used

| Category | Technologies |
|:--------:|:------------:|
| Backend | Flask |
| NLP | spaCy, HuggingFace Transformers (RoBERTa) |
| Skill Matching | thefuzz (fuzzy string matching) |
| Embedding Similarity | Cosine Similarity (scikit-learn) |
| File Parsing | PyPDF2 (for PDFs), python-docx (for DOCX files) |
| Course Recommendations | JSON Dataset |
| Frontend | HTML, CSS (via Flask Templates) |

📚 Model & Techniques

- **Text Preprocessing**: Lowercasing, Lemmatization (spaCy).
- **Semantic Similarity**: 
  - RoBERTa Embeddings for resume and job description.
  - Cosine similarity to measure closeness.
- **Skill Extraction**:
  - Fuzzy Matching (Partial Ratio) with thresholding for flexible matching.
- **Course Recommendation**:
  - Fuzzy matched missing skills to an online courses dataset.

⚙️ How It Works

1. **Candidate Side**:
   - Upload Resume (.pdf/.docx)
   - Enter Job Description & Required Skills
   - Get: 
     - **Semantic Score** (Resume vs Job Description)
     - **Skill Match Score**
     - **Final Fit Score** (60% semantic + 40% skills)
     - **Matched Skills**, **Missing Skills**
     - **Recommended Courses** for missing skills.

2. **HR Side**:
   - Upload **multiple resumes** for a job role.
   - Enter Job Description & Required Skills.
   - Get resumes **ranked** based on Final Fit Score for smart hiring decisions.

🛤️ Project Structure

```
/templates
    ├── home.html
    ├── index.html
    ├── result.html
    ├── multi_upload.html
    └── multi_result.html
/static
    └── styles.css
/data
    └── online_courses_dataset.json
app.py
requirements.txt
README.md
```

---

🧩 Setup Instructions

1. Clone the Repository
   ```bash
   git clone https://github.com/AbhijitPalve1506/smart-resume-analyzer.git
   cd smart-resume-analyzer
   ```

2. Create virtual environment (use python version 11 or 12)
   ```bash
   python -m venv "your_venv_name"
   ```

3. Install Requirements
   ```bash
   pip install -r requirements.txt
   ```

4. Download SpaCy Model
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Run the App
   ```bash
   python app.py
   ```

6. Open your browser and navigate to
   ```
   http://127.0.0.1:5000/
   ```

## 📌 Important Notes
- Ensure **CUDA** is installed if you want faster RoBERTa embeddings with GPU. (Otherwise, it falls back to CPU.)
- The course recommendation is based on a **fuzzy matching threshold of 80%**.
- The current version uses **basic HTML templates** — can be enhanced later with Bootstrap/Tailwind for UI improvements.

✨ Thank You for checking out Smart Resume Analyzer!

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
