# app.py
# Enhanced Streamlit Application for Resume Screening

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import re

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

def classify_and_summarize(resume, job_description):
    original_resume = resume
    resume = normalize_text(resume)
    job_description = normalize_text(job_description)
    input_text = f"resume: {resume} [sep] job: {job_description}"
    
    inputs = bert_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    prediction = np.argmax(probabilities)
    
    confidence_threshold = 0.95
    if probabilities[prediction] < confidence_threshold:
        suitability = "Uncertain"
        warning = f"Low confidence: {probabilities[prediction]:.4f}"
    else:
        suitability = "Relevant" if prediction == 1 else "Irrelevant"
        warning = None
    
    exp_warning = check_experience_mismatch(original_resume, job_description)
    if exp_warning and suitability == "Relevant":
        suitability = "Uncertain"
        warning = exp_warning if not warning else f"{warning}; {exp_warning}"
    
    prompt = f"summarize: {resume}"
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
    
    return suitability, summary, warning

# Streamlit interface
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="centered")

# Introduction
st.markdown("""
    <h1 style='text-align: center; color: #2E4053;'>Resume Screening Application</h1>
    <p style='text-align: center; color: #566573;'>
        Welcome to our AI-powered resume screening tool! This app evaluates resumes against job descriptions to determine suitability, providing a concise summary of key skills and experience. Built with advanced natural language processing, it ensures accurate and efficient screening.
    </p>
""", unsafe_allow_html=True)

# Instructions and Guidelines
with st.expander("📋 How to Use the App", expanded=False):
    st.markdown("""
        **Instructions**:
        - Enter the candidate's resume in the first text box, listing skills and experience (e.g., "Expert in python, machine learning, 4 years experience").
        - Enter the job description in the second text box, specifying required skills and experience (e.g., "Data scientist requires python, machine learning, 3 years+").
        - Click **Analyze** to get the suitability and summary.
        - Use the **Reset** button to clear inputs and start over.

        **Guidelines**:
        - Use clear, comma-separated lists for skills (e.g., "python, sql, pandas").
        - Include experience in years (e.g., "4 years experience") or as "senior" for senior-level roles.
        - Avoid ambiguous phrases; be specific about skills and requirements.
    """)

# Classification Criteria
with st.expander("ℹ️ Classification Criteria", expanded=False):
    st.markdown("""
        The app classifies resumes based on:
        - **Skill Overlap**: At least 70% of the job’s required skills must match the resume’s skills.
        - **Experience Match**: The resume’s experience (in years or seniority) must meet or exceed the job’s requirement.
        
        **Outcomes**:
        - **Relevant**: High skill overlap and sufficient experience, with strong confidence (≥95%).
        - **Irrelevant**: Low skill overlap or insufficient experience, with strong confidence.
        - **Uncertain**: Borderline confidence (<95%) or experience mismatch (e.g., resume has 2 years, job requires 3 years+).
        
        **Note**: An experience mismatch warning is shown if the resume’s experience is below the job’s requirement, even if skills match.
    """)

# Input form
st.markdown("### 📝 Enter Details")
col1, col2 = st.columns([1, 1])
with col1:
    resume = st.text_area("Resume", value="Expert in python, machine learning, tableau, 4 years experience", height=100, key="resume")
with col2:
    job_description = st.text_area("Job Description", value="Data scientist requires python, machine learning, 3 years+", height=100, key="job_description")

# Buttons
col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    analyze_clicked = st.button("Analyze", type="primary")
with col_btn2:
    reset_clicked = st.button("Reset")

# Handle reset
if reset_clicked:
    st.session_state.resume = ""
    st.session_state.job_description = ""
    st.experimental_rerun()

# Handle analysis
if analyze_clicked:
    if resume.strip() and job_description.strip():
        with st.spinner("Analyzing resume..."):
            suitability, summary, warning = classify_and_summarize(resume, job_description)
        st.success("Analysis completed! 🎉")
        st.markdown("### 📊 Results")
        st.markdown(f"**Suitability**: {suitability}", unsafe_allow_html=True)
        st.markdown(f"**Summary**: {summary}", unsafe_allow_html=True)
        if warning:
            st.warning(f"**Warning**: {warning}")
    else:
        st.error("Please enter both a resume and job description.")