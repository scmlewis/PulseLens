# app.py
# Optimized Streamlit Application for Resume Screening with Multiple Resumes

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Resume Screening Assistant for Data/Tech", page_icon="📄", layout="wide")

# Set sidebar width and make uncollapsible
st.markdown("""
    <style>
    .css-1d391kg {  /* Sidebar */
        width: 350px !important;
    }
    [data-testid="stSidebarCollapseButton"] {  /* Hide toggle button */
        display: none !important;
    }
    .stSidebar {  /* Ensure sidebar visibility */
        min-width: 350px !important;
        visibility: visible !important;
    }
    [data-testid="stExpander"] summary {  /* Expander headers */
        font-size: 26px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1) !important;
        white-space: nowrap !important;
    }
    .st-expander-content p {  /* Expander body text */
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Skills list (79 skills from Application_Demo.ipynb)
skills_list = [
    'python', 'sql', 'c++', 'java', 'tableau', 'machine learning', 'data analysis',
    'business intelligence', 'r', 'tensorflow', 'pandas', 'spark', 'scikit-learn', 'aws',
    'javascript', 'scala', 'go', 'ruby', 'pytorch', 'keras', 'deep learning', 'nlp',
    'computer vision', 'azure', 'gcp', 'docker', 'kubernetes', 'hadoop', 'kafka',
    'airflow', 'power bi', 'matplotlib', 'seaborn', 'plotly', 'ggplot', 'mysql',
    'postgresql', 'mongodb', 'redis', 'git', 'linux', 'api', 'rest',
    'rust', 'kotlin', 'typescript', 'julia', 'snowflake', 'bigquery', 'cassandra',
    'neo4j', 'hugging face', 'langchain', 'onnx', 'xgboost', 'terraform', 'ansible',
    'jenkins', 'gitlab ci', 'qlik', 'looker', 'd3 js', 'blockchain', 'quantum computing',
    'cybersecurity', 'project management', 'technical writing', 'business analysis',
    'agile methodologies', 'communication', 'team leadership',
    'databricks', 'synapse', 'delta lake', 'streamlit', 'fastapi', 'graphql', 'mlflow', 'kedro'
]

# Precompile regex for skills matching (optimized for single pass)
skills_pattern = re.compile(r'\b(' + '|'.join(re.escape(skill) for skill in skills_list) + r')\b', re.IGNORECASE)

# Helper functions for CV parsing
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from Word document: {str(e)}")
        return ""

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or Word (.docx) document.")
        return ""

# Helper functions for analysis
def normalize_text(text):
    text = text.lower()
    # Remove underscores, hyphens, and specific phrases, replacing with empty string
    text = re.sub(r'_|-|,\s*collaborated in agile teams|,\s*developed solutions for|,\s*led projects involving|,\s*designed applications with|,\s*built machine learning models for|,\s*implemented data pipelines for|,\s*deployed cloud-based solutions|,\s*optimized workflows for|,\s*contributed to data-driven projects', '', text)
    return text

def check_experience_mismatch(resume, job_description):
    resume_match = re.search(r'(\d+)\s*years?|senior', resume.lower())
    # Allow optional words like "experience" between "years" and "+"
    job_match = re.search(r'(\d+)\s*years?(?:\s+\w+)*\+|senior\+', job_description.lower())
    if resume_match and job_match:
        resume_years = resume_match.group(0)
        job_years = job_match.group(0)
        # Handle resume years
        if 'senior' in resume_years:
            resume_num = 10
        else:
            resume_num = int(resume_match.group(1))
        # Handle job years
        if 'senior+' in job_years:
            job_num = 10 
        else:
            job_num = int(job_match.group(1))
        if resume_num < job_num:
            return f"Experience mismatch: Resume has {resume_years}, job requires {job_years}"
    return None

def validate_input(text, is_resume=True):
    if not text.strip() or len(text.strip()) < 10:
        return "Input is too short (minimum 10 characters)."
    text_normalized = normalize_text(text)
    if is_resume and not skills_pattern.search(text_normalized):
        return "Please include at least one data/tech skill (e.g., python, sql, databricks)."
    if is_resume and not re.search(r'\d+\s*year(s)?|senior', text.lower()):
        return "Please include experience (e.g., '3 years experience' or 'senior')."
    return None

@st.cache_resource
def load_models():
    bert_model_path = 'scmlewis/bert-finetuned-isom5240'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    device = torch.device('cpu')  # CPU for lightweight deployment
    bert_model.to(device)
    t5_model.to(device)
    bert_model.eval()
    t5_model.eval()
    return bert_tokenizer, bert_model, t5_tokenizer, t5_model, device

@st.cache_data
def tokenize_inputs(resumes, job_description, _bert_tokenizer, _t5_tokenizer):
    """Precompute tokenized inputs for BERT and T5."""
    job_description_norm = normalize_text(job_description)
    bert_inputs = [f"resume: {normalize_text(resume)} [sep] job: {job_description_norm}" for resume in resumes]
    bert_tokenized = _bert_tokenizer(bert_inputs, return_tensors='pt', padding=True, truncation=True, max_length=64)
    
    t5_inputs = []
    for resume in resumes:
        prompt = re.sub(r'\b[Cc]\+\+\b', 'c++', resume)
        prompt_normalized = normalize_text(prompt)
        t5_inputs.append(f"summarize: {prompt_normalized}")
    t5_tokenized = _t5_tokenizer(t5_inputs, return_tensors='pt', padding=True, truncation=True, max_length=64)
    
    return bert_tokenized, t5_inputs, t5_tokenized

@st.cache_data
def extract_skills(text):
    """Extract skills from text in a single pass."""
    text_normalized = normalize_text(text)
    text_normalized = re.sub(r'[,_-]', ' ', text_normalized)
    found_skills = skills_pattern.findall(text_normalized)
    return set(found_skills)

@st.cache_data
def classify_and_summarize_batch(resume, job_description, _bert_tokenized, _t5_input, _t5_tokenized, _job_skills_set):
    """Process one resume at a time to reduce CPU load with a timeout."""
    _, bert_model, t5_tokenizer, t5_model, device = st.session_state.models
    start_time = time.time()
    timeout = 60  # Timeout after 60 seconds
    
    try:
        bert_tokenized = {k: v.to(device) for k, v in _bert_tokenized.items()}
        with torch.no_grad():
            # BERT inference
            bert_start = time.time()
            outputs = bert_model(**bert_tokenized)
            if time.time() - bert_start > timeout:
                raise TimeoutError("BERT inference timed out")
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)
        
        confidence_threshold = 0.85
        
        t5_tokenized = {k: v.to(device) for k, v in _t5_tokenized.items()}
        with torch.no_grad():
            # T5 inference
            t5_start = time.time()
            t5_outputs = t5_model.generate(
                t5_tokenized['input_ids'],
                attention_mask=t5_tokenized['attention_mask'],
                max_length=30,
                min_length=8,
                num_beams=2,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True
            )
            if time.time() - t5_start > timeout:
                raise TimeoutError("T5 inference timed out")
        summaries = [t5_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in t5_outputs]
        summaries = [re.sub(r'\s+', ' ', summary).strip() for summary in summaries]
        
        prob, pred, summary, t5_input = probabilities[0], predictions[0], summaries[0], _t5_input
        resume_skills_set = extract_skills(resume)
        skill_overlap = len(_job_skills_set.intersection(resume_skills_set)) / len(_job_skills_set) if _job_skills_set else 0

        if skill_overlap < 0.4:
            suitability = "Irrelevant"
            warning = "Skills are irrelevant"
        else:
            exp_warning = check_experience_mismatch(resume, job_description)
            if exp_warning:
                suitability = "Uncertain"
                warning = exp_warning
            else:
                if prob[pred] < confidence_threshold:
                    suitability = "Uncertain"
                    warning = f"Low confidence: {prob[pred]:.4f}"
                else:
                    suitability = "Relevant" if skill_overlap >= 0.5 else "Irrelevant"
                    warning = "Skills are not a strong match" if suitability == "Irrelevant" else None
        
        skills = list(set(skills_pattern.findall(t5_input)))  # Deduplicate skills
        exp_match = re.search(r'\d+\s*years?|senior', resume.lower())
        if skills and exp_match:
            summary = f"{', '.join(skills)} proficiency, {exp_match.group(0)} experience"
        else:
            summary = f"{exp_match.group(0) if exp_match else 'unknown'} experience"
        
        result = {
            "Suitability": suitability,
            "Data/Tech Related Skills Summary": summary,
            "Warning": warning or "None"
        }
        
        return result
    except TimeoutError as e:
        st.warning(f"Skipped processing for resume due to timeout: {str(e)}")
        return {
            "Suitability": "Error",
            "Data/Tech Related Skills Summary": "Processing timed out",
            "Warning": str(e)
        }
    except Exception as e:
        st.error(f"Error during inference for resume: {str(e)}")
        return {
            "Suitability": "Error",
            "Data/Tech Related Skills Summary": "Failed to process",
            "Warning": str(e)
        }

@st.cache_data
def generate_skill_pie_chart(resumes):
    skill_counts = {}
    total_resumes = len([r for r in resumes if r.strip()])
    
    if total_resumes == 0:
        return None
    
    # Count skills that appear in resumes
    for resume in resumes:
        if resume.strip():
            resume_lower = normalize_text(resume)
            found_skills = skills_pattern.findall(resume_lower)
            for skill in found_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    if not skill_counts:
        return None
    
    labels = list(skill_counts.keys())
    sizes = [(count / sum(skill_counts.values())) * 100 for count in skill_counts.values()]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10})
    ax.axis('equal')
    plt.title("Skill Frequency Across Resumes", fontsize=12, color='#007BFF', pad=10)
    return fig

def render_sidebar():
    """Render sidebar content."""
    with st.sidebar:
        st.markdown("""
            <h1 style='text-align: center; color: #007BFF; font-size: 32px; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>💻 Resume Screening Assistant for Data/Tech</h1>
            <p style='text-align: center; font-size: 16px; margin-top: 0;'>
                Welcome to our AI-powered resume screening tool, specialized for data science and tech roles! This app evaluates multiple resumes against a single job description to determine suitability, providing concise summaries of key data and tech skills and experience. Built with advanced natural language processing, it ensures accurate and efficient screening for technical positions. <br><br><strong>Note:</strong> Performance may vary due to server load on free CPU instances.
            </p>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 How to Use the App", expanded=True):
            st.markdown("""
                **Instructions**:
                - Upload a PDF or Word (.docx) CV or manually enter up to 5 candidate resumes in the text boxes, listing data/tech skills and experience (e.g., "Expert in python, databricks, 6 years experience").
                - Enter the job description, specifying required skills and experience (e.g., "Data engineer requires python, spark, 5 years+").
                - Click **Analyze** to evaluate all non-empty resumes (at least one required).
                - Use **Add Resume** or **Remove Resume** to adjust the number of resume fields (1–5).
                - Use the **Reset** button to clear all inputs and results.
                - Download results as a CSV file for record-keeping.
                - View the skill frequency pie chart to see skill distribution across resumes.
                - Example test cases:
                  - **Test Case 1**: Resumes like "Expert in python, machine learning, tableau, 4 years experience" against "Data scientist requires python, machine learning, 3 years+".
                  - **Test Case 2**: Resumes like "Skilled in databricks, spark, python, 6 years experience" against "Data engineer requires python, spark, 5 years+".

                **Guidelines**:
                - Use comma-separated skills from a comprehensive list including python, sql, databricks, etc. (79 skills supported).
                - Include experience in years (e.g., "3 years experience" or "1 year experience") or as "senior".
                - Focus on data/tech skills for accurate summarization.
                - Resumes with only irrelevant skills (e.g., sales, marketing) will be classified as "Irrelevant".
                - If uploading a CV, ensure it’s a text-based PDF or Word document (scanned PDFs may not work).
            """)
        with st.expander("ℹ️ Classification Criteria", expanded=True):
            st.markdown("""
                The app classifies resumes based on:
                - **Skill Overlap**: The resume’s data/tech skills are compared to the job’s requirements. A skill overlap below 40% results in an "Irrelevant" classification.
                - **Model Confidence**: A finetuned BERT model evaluates skill relevance. If confidence is below 85%, the classification is "Uncertain".
                - **Experience Match**: The resume’s experience (in years or seniority) must meet or exceed the job’s requirement.

                **Outcomes**:
                - **Relevant**: Skill overlap ≥ 50%, sufficient experience, and high model confidence (≥85%).
                - **Irrelevant**: Skill overlap < 40% or high confidence in low skill relevance.
                - **Uncertain**: Skill overlap ≥ 50% but experience mismatch (e.g., resume has 2 years, job requires 5 years+), or low model confidence (<85%).

                **Note**: An experience mismatch warning is shown if the resume’s experience is below the job’s requirement, overriding the skill overlap and confidence to classify as Uncertain.
            """)

def main():
    """Main function to run the Streamlit app for resume screening."""
    # Render sidebar
    render_sidebar()

    # Initialize session state
    if 'resumes' not in st.session_state:
        st.session_state.resumes = ["Expert in python, machine learning, tableau, 4 years experience", "", ""]
    if 'input_job_description' not in st.session_state:
        st.session_state.input_job_description = "Data scientist requires python, machine learning, 3 years+"
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'valid_resumes' not in st.session_state:
        st.session_state.valid_resumes = []
    if 'models' not in st.session_state:
        st.session_state.models = None

    # Resume inputs with file upload and manual text input
    st.markdown("### 📝 Enter Resumes")
    for i in range(len(st.session_state.resumes)):
        st.markdown(f"**Resume {i+1}**")
        uploaded_file = st.file_uploader(f"Upload CV (PDF or Word) for Resume {i+1}", type=['pdf', 'docx'], key=f"file_upload_{i}")
        
        if uploaded_file is not None:
            extracted_text = extract_text_from_file(uploaded_file)
            if extracted_text:
                st.session_state.resumes[i] = extracted_text
            else:
                st.session_state.resumes[i] = ""

        st.session_state.resumes[i] = st.text_area(
            f"Enter or edit resume text",
            value=st.session_state.resumes[i],
            height=100,
            key=f"resume_{i}",
            placeholder="e.g., Expert in python, sql, 3 years experience"
        )
        validation_error = validate_input(st.session_state.resumes[i], is_resume=True)
        if validation_error and st.session_state.resumes[i].strip():
            st.warning(f"Resume {i+1}: {validation_error}")

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

    # Job description input
    st.markdown("### 📋 Enter Job Description")
    job_description = st.text_area(
        "Job Description",
        value=st.session_state.input_job_description,
        height=100,
        key="job_description",
        placeholder="e.g., Data scientist requires python, sql, 3 years+"
    )
    validation_error = validate_input(job_description, is_resume=False)
    if validation_error and job_description.strip():
        st.warning(f"Job Description: {validation_error}")

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
        st.session_state.valid_resumes = []
        st.rerun()

    # Handle analysis with early validation and lazy model loading
    if analyze_clicked:
        # Early validation of inputs
        valid_resumes = []
        for i, resume in enumerate(st.session_state.resumes):
            validation_error = validate_input(resume, is_resume=True)
            if not validation_error and resume.strip():
                valid_resumes.append(resume)
            elif validation_error and resume.strip():
                st.warning(f"Resume {i+1}: {validation_error}")

        validation_error = validate_input(job_description, is_resume=False)
        if validation_error and job_description.strip():
            st.warning(f"Job Description: {validation_error}")

        if valid_resumes and job_description.strip():
            # Load models only when needed
            if st.session_state.models is None:
                with st.spinner("Loading models, please wait..."):
                    st.session_state.models = load_models()

            st.session_state.results = []
            st.session_state.valid_resumes = valid_resumes
            total_steps = len(valid_resumes)
            
            with st.spinner("Analyzing resumes..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Preparing inputs...")
                
                # Retrieve tokenizers from st.session_state.models
                bert_tokenizer, bert_model, t5_tokenizer, t5_model, device = st.session_state.models
                
                # Precompute tokenized inputs and job skills
                bert_tokenized, t5_inputs, t5_tokenized = tokenize_inputs(valid_resumes, job_description, bert_tokenizer, t5_tokenizer)
                job_skills_set = extract_skills(job_description)
                
                status_text.text("Classifying and summarizing resumes...")
                results = []
                for i, (resume, bert_tok, t5_in, t5_tok) in enumerate(zip(valid_resumes, bert_tokenized['input_ids'], t5_inputs, t5_tokenized['input_ids'])):
                    status_text.text(f"Processing Resume {i+1}/{total_steps}: {resume[:50]}...")
                    result = classify_and_summarize_batch(
                        resume,
                        job_description,
                        {'input_ids': bert_tok.unsqueeze(0), 'attention_mask': bert_tokenized['attention_mask'][i].unsqueeze(0)},
                        t5_in,
                        {'input_ids': t5_tok.unsqueeze(0), 'attention_mask': t5_tokenized['attention_mask'][i].unsqueeze(0)},
                        job_skills_set
                    )
                    result["Resume"] = f"Resume {i+1}"
                    results.append(result)
                    progress_bar.progress((i + 1) / total_steps)
                
                st.session_state.results = results
                
                status_text.empty()
                progress_bar.empty()
                st.success("Analysis completed! 🎉")
        else:
            st.error("Please enter at least one valid resume and a job description.")

    # Display results
    if st.session_state.results:
        st.markdown("### 📊 Results")
        st.table(st.session_state.results)
        
        csv_buffer = io.StringIO()
        csv_buffer.write("Resume Number,Resume Text,Job Description,Suitability,Summary,Warning\n")
        for i, result in enumerate(st.session_state.results):
            resume_text = st.session_state.valid_resumes[i].replace('"', '""').replace('\n', ' ')
            job_text = job_description.replace('"', '""').replace('\n', ' ')
            csv_buffer.write(f'"{result["Resume"]}","{resume_text}","{job_text}","{result["Suitability"]}","{result["Data/Tech Related Skills Summary"]}","{result["Warning"]}"\n')
        st.download_button("Download Results", csv_buffer.getvalue(), file_name="resume_analysis.csv", mime="text/csv")
        
        with st.expander("📈 Skill Frequency Across Resumes", expanded=False):
            if st.session_state.valid_resumes:
                fig = generate_skill_pie_chart(st.session_state.valid_resumes)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("No recognized data/tech skills found in the resumes.")
            else:
                st.write("No valid resumes to analyze.")

if __name__ == "__main__":
    # When this module is run directly, call the main function.
    main()