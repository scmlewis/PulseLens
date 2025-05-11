# app.py
# Stage 5: Streamlit Application for Resume Screening

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import re

# Set page configuration
st.set_page_config(page_title="Resume Screening Application", page_icon="📄")

# Title and description
st.title("Resume Screening Application")
st.markdown("""
This application classifies a resume-job pair as **Relevant** or **Irrelevant** and generates a concise summary of the resume's skills.

**Classification Criteria**:
- **Skill Overlap**: At least 80% of the job's required skills must be in the resume.
- **Experience Match**: The resume's experience must meet or exceed the job's requirement.
- **Outcome**: Relevant if both conditions are met; otherwise, Irrelevant.
""")

# Load models
@st.cache_resource
def load_models():
    bert_model_path = 'scmlewis/bert-finetuned-isom5240'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)
    t5_generator = pipeline('text2text-generation', model='t5-small')
    return bert_tokenizer, bert_model, t5_generator

bert_tokenizer, bert_model, t5_generator = load_models()

# Input fields
st.subheader("Enter Resume and Job Description")
resume = st.text_area("Resume", placeholder="e.g., Skilled in Python, SQL, 3 years experience")
job_description = st.text_area("Job Description", placeholder="e.g., Data analyst requires Python, SQL, 3 years+")

# Process inputs
if st.button("Screen Resume"):
    if resume and job_description:
        # Classification
        input_text = f"resume: {resume} [sep] job: {job_description}"
        inputs = bert_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        suitability = "Relevant" if outputs.logits.argmax().item() == 1 else "Irrelevant"
        
        # Summary
        simplified_resume = re.sub(r'(versed in leveraging|designed applications for|created solutions with|led projects involving|collaborated in agile teams over)', 'proficient in', resume).strip()
        simplified_resume = re.sub(r'\s+', ' ', simplified_resume)
        prompt = f"summarize: {simplified_resume}"
        summary = t5_generator(
            prompt,
            max_length=20,
            min_length=5,
            num_beams=15,
            no_repeat_ngram_size=3,
            length_penalty=0.5,
            early_stopping=True
        )[0]['generated_text']
        
        # Display results
        st.subheader("Results")
        st.write(f"**Suitability**: {suitability}")
        st.write(f"**Summary**: {summary}")
    else:
        st.error("Please enter both a resume and a job description.")