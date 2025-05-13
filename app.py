# app.py
# Enhanced Streamlit Application for Resume Screening with Multiple Resumes (Group36, ISOM5240 Topic 18)

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import re
import io

# Set page config as the first Streamlit command
st.set_page_config(page_title="Resume Screening Assistant for Data/Tech", page_icon="📄", layout="centered")

# Set sidebar width
st.markdown("""
    <style>
    .css-1d391kg {  /* Sidebar class in Streamlit */
        width: 350px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    bert_model_path = 'scmlewis/bert-finetuned-isom5240'
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    device = torch.device('cpu')  # CPU for lightweight deployment
    bert_model.to(device)
    t5_model.to(device)
    bert_model.eval()
    t5_model.eval()
    return bert_tokenizer, bert_model, t5_tokenizer, t5_model, device

bert_tokenizer, bert_model, t5_tokenizer, t5_model, device = load_models()

# Helper functions
def normalize_text(text):
    text = text.lower()
    text = re.sub(r',\s*collaborated in agile teams|,\s*developed solutions for|,\s*led projects involving|,\s*designed applications with|,\s*built machine learning models for|,\s*implemented data pipelines for|,\s*deployed cloud-based solutions|,\s*optimized workflows for|,\s*contributed to data-driven projects', '', text)
    return text

def check_experience_mismatch(resume, job_description):
    resume_match = re.search(r'(\d+)\s*years?|senior', resume.lower())
    job_match = re.search(r'(\d+)\s*years?\+|senior\+', job_description.lower())
    if resume_match and job_match:
        resume_years = resume_match.group(0)
        job_years = job_match.group(0)
        resume_num = 10 if 'senior' in resume_years else int(resume_years.split()[0])
        job_num = 10 if 'senior' in job_years else int(job_years.split()[0])
        if resume_num < job_num:
            return f"Experience mismatch: Resume has {resume_years}, job requires {job_years}"
    return None

def validate_input(text, is_resume=True):
    if not text.strip() or len(text.strip()) < 10:
        return "Input is too short (minimum 10 characters)."
    if not re.search(r'\b(python|sql|pandas|java|c\+\+|machine\s*learning|tableau)\b', text.lower()):
        return "Please include at least one data/tech skill (e.g., python, sql)."
    if is_resume and not re.search(r'\d+\s*years|senior', text.lower()):
        return "Please include experience (e.g., '4 years experience' or 'senior')."
    return None

def classify_and_summarize_batch(resumes, job_description):
    job_description = normalize_text(job_description)
    inputs = [f"resume: {normalize_text(resume)} [sep] job: {job_description}" for resume in resumes]
    tokenized = bert_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=128)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    
    with torch.no_grad():
        outputs = bert_model(**tokenized)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)
    
    confidence_threshold = 0.85  # Adjusted to 85% for testing
    results = []
    for i, (resume, prob, pred) in enumerate(zip(resumes, probabilities, predictions)):
        if prob[pred] < confidence_threshold:
            suitability = "Uncertain"
            warning = f"Low confidence: {prob[pred]:.4f}"
        else:
            suitability = "Relevant" if pred == 1 else "Irrelevant"
            warning = None
        
        exp_warning = check_experience_mismatch(resume, job_description)
        if exp_warning and suitability == "Relevant":
            suitability = "Uncertain"
            warning = exp_warning if not warning else f"{warning}; {exp_warning}"
        
        prompt = f"summarize: {normalize_text(resume)}"
        inputs = t5_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs['input_ids'],
                max_length=18,
                min_length=8,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=3.0,
                early_stopping=True
            )
        
        summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summary = re.sub(r'with\s*(sql|pandas|java|c\+\+|python|machine\s*learning|tableau|\d+\s*years)\s*(and\s*\1)?', '', summary).strip()
        summary = re.sub(r'\b(skilled in|proficient in|expert in|versed in|experienced in|specialized in|accomplished in|trained in)\b', '', summary).strip()
        summary = re.sub(r'\s*and\s*(sql|pandas|java|c\+\+|python|machine\s*learning|tableau|\d+\s*years)', '', summary).strip()
        summary = re.sub(r'experience\s*(of|and)\s*experience', 'experience', summary).strip()
        summary = re.sub(r'years\s*years', 'years', summary).strip()
        skills = re.findall(r'\b(python|sql|pandas|java|c\+\+|machine\s*learning|tableau)\b', prompt.lower())
        exp_match = re.search(r'\d+\s*years|senior', resume.lower())
        if skills and exp_match:
            summary = f"{', '.join(skills)} proficiency, {exp_match.group(0)} experience"
        else:
            summary = f"{exp_match.group(0) if exp_match else 'unknown'} experience"
        
        results.append({
            "Resume": f"Resume {st.session_state.resumes.index(resume)+1}",
            "Suitability": f"✅ {suitability}" if suitability == "Relevant" else f"❌ {suitability}" if suitability == "Irrelevant" else f"❓ {suitability}",
            "Data/Tech Related Skills Summary": summary,
            "Warning": warning or "None"
        })
    
    return results

# Streamlit interface
# Sidebar with Instructions and Criteria
with st.sidebar:
    with st.expander("📋 How to Use the App", expanded=True):
        st.markdown("""
            **Instructions**:
            - Enter up to 5 candidate resumes in the text boxes, listing data/tech skills and experience (e.g., "Expert in python, machine learning, 4 years experience").
            - Enter the job description in the provided text box, specifying required skills and experience (e.g., "Data scientist requires python, machine learning, 3 years+").
            - Click **Analyze** to evaluate all non-empty resumes (at least one required).
            - Use **Add Resume** or **Remove Resume** to adjust the number of resume fields.
            - Use the **Reset** button to clear all inputs and results.
            - Download results as a CSV file for record-keeping.

            **Guidelines**:
            - Use clear, comma-separated lists for skills (e.g., "python, sql, pandas").
            - Include experience in years (e.g., "4 years experience") or as "senior" for senior-level roles.
            - Focus on data science and tech skills, as the app summarizes only these (e.g., python, sql, machine learning).
        """)
    with st.expander("ℹ️ Classification Criteria", expanded=True):
        st.markdown("""
            The app classifies resumes based on:
            - **Skill Overlap**: At least 70% of the job’s required data/tech skills must match the resume’s skills.
            - **Experience Match**: The resume’s experience (in years or seniority) must meet or exceed the job’s requirement.
            
            **Outcomes**:
            - **Relevant** ✅: High skill overlap and sufficient experience, with strong confidence (≥85%).
            - **Irrelevant** ❌: Low skill overlap or insufficient experience, with strong confidence.
            - **Uncertain** ❓: Borderline confidence (<85%) or experience mismatch (e.g., resume has 2 years, job requires 3 years+).
            
            **Note**: An experience mismatch warning (⚠️) is shown if the resume’s experience is below the job’s requirement, even if skills match.
        """)

# Introduction
st.markdown("""
    <h1 style='text-align: center; color: #007BFF; font-size: 36px; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);'>💻 Resume Screening Assistant for Data/Tech 📊</h1>
    <p style='text-align: center; color: #566573;'>
        Welcome to our AI-powered resume screening tool, specialized for data science and tech roles! This app evaluates multiple resumes against a single job description to determine suitability, providing concise summaries of key data and tech skills and experience. Built with advanced natural language processing, it ensures accurate and efficient screening for technical positions.
    </p>
""", unsafe_allow_html=True)

# Input form
st.markdown("### 📝 Enter Resumes")
# Initialize resumes and results in session state
if 'resumes' not in st.session_state:
    st.session_state.resumes = ["Expert in python, machine learning, tableau, 4 years experience", "", ""]
if 'input_job_description' not in st.session_state:
    st.session_state.input_job_description = "Data scientist requires python, machine learning, 3 years+"
if 'results' not in st.session_state:
    st.session_state.results = []

# Resume inputs
for i in range(len(st.session_state.resumes)):
    st.session_state.resumes[i] = st.text_area(f"Resume {i+1}", value=st.session_state.resumes[i], height=100, key=f"resume_{i}")
    # Real-time validation for resumes
    validation_error = validate_input(st.session_state.resumes[i], is_resume=True)
    if validation_error and st.session_state.resumes[i].strip():
        st.warning(f"Resume {i+1}: {validation_error}")

# Job description input
st.markdown("### 📋 Enter Job Description")
job_description = st.text_area("Job Description", value=st.session_state.input_job_description, height=100, key="job_description")
# Real-time validation for job description
validation_error = validate_input(job_description, is_resume=False)
if validation_error and job_description.strip():
    st.warning(f"Job Description: {validation_error}")

# Add/Remove resume buttons
col_add, col_remove, _ = st.columns([1, 1, 3])
with col_add:
    if st.button("Add Resume") and len(st.session_state.resumes) < 5:
        st.session_state.resumes.append("")
        st.rerun()
with col_remove:
    if st.button("Remove Resume") and len(st.session_state.resumes) > 1:
        st.session_state.resumes.pop()
        st.rerun()

# Analyze and Reset buttons
col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    analyze_clicked = st.button("Analyze", type="primary")
with col_btn2:
    reset_clicked = st.button("Reset")

# Handle reset
if reset_clicked:
    st.session_state.resumes = ["", "", ""]
    st.session_state.input_job_description = ""
    st.session_state.results = []
    st.rerun()

# Handle analysis
if analyze_clicked:
    valid_resumes = [resume for resume in st.session_state.resumes if resume.strip()]
    if valid_resumes and job_description.strip():
        st.session_state.results = []  # Clear previous results
        total_steps = len(valid_resumes) + 1  # BERT batch + T5 per resume
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Batch classification
        status_text.text("Classifying resumes (batch processing)...")
        results = classify_and_summarize_batch(valid_resumes, job_description)
        progress_bar.progress(1 / total_steps)
        
        st.session_state.results = results
        
        status_text.empty()
        progress_bar.empty()
        st.success("Analysis completed! 🎉")
    
    else:
        st.error("Please enter at least one resume and a job description.")

# Display results
if st.session_state.results:
    st.markdown("### 📊 Results")
    st.table(st.session_state.results)
    
    # Download results as CSV
    csv_buffer = io.StringIO()
    csv_buffer.write("Resume Number,Resume Text,Job Description,Suitability,Summary,Warning\n")
    for i, result in enumerate(st.session_state.results):
        resume_text = valid_resumes[i].replace('"', '""').replace('\n', ' ')
        job_text = job_description.replace('"', '""').replace('\n', ' ')
        suitability = result["Suitability"].replace('✅ ', '').replace('❌ ', '').replace('❓ ', '')
        csv_buffer.write(f'"{result["Resume"]}","{resume_text}","{job_text}","{suitability}","{result["Data/Tech Related Skills Summary"]}","{result["Warning"]}"\n')
    st.download_button("Download Results", csv_buffer.getvalue(), file_name="resume_analysis.csv", mime="text/csv")