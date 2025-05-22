# app.py
# Optimized Streamlit Application for Resume Screening with Multiple Resumes and Professional Theme

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import re
import io
import matplotlib.pyplot as plt

# Set page config as the first Streamlit command
st.set_page_config(page_title="Resume Screening Assistant for Data/Tech", page_icon="📄", layout="wide")

# Apply simplified custom CSS for a professional theme inspired by Databricks
st.markdown("""
    <style>
    /* General App Styling */
    .stApp {
        background-color: #F5F5F5; /* Light gray background */
        font-family: 'Arial', sans-serif;
    }

    /* Header Banner */
    .header-banner {
        background-color: #FF3621; /* Databricks orange */
        color: #FFFFFF;
        padding: 15px;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .header-banner h1 {
        margin: 0;
        font-size: 32px;
    }

    /* Sidebar Styling */
    .css-1d391kg {  /* Sidebar */
        width: 350px !important;
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    [data-testid="stSidebarCollapseButton"] {  /* Hide toggle button */
        display: none !important;
    }
    .stSidebar {
        min-width: 350px !important;
        visibility: visible !important;
    }
    .stSidebar h1 {
        color: #FF3621; /* Databricks orange */
        font-size: 32px;
        margin-bottom: 10px;
    }
    .stSidebar p {
        color: #4A4A4A; /* Databricks gray */
        font-size: 16px;
    }

    /* Expander Styling */
    [data-testid="stExpander"] summary {
        font-size: 26px !important;
        font-weight: bold !important;
        color: #FF3621 !important;
        white-space: nowrap !important;
    }
    .st-expander-content p {
        font-size: 12px !important;
        color: #4A4A4A !important;
    }

    /* Main Content Styling */
    h2, h3 {
        color: #FF3621; /* Databricks orange */
        font-weight: bold;
        margin-top: 20px;
    }

    /* Input Fields */
    .stTextInput > label {
        color: #4A4A4A; /* Databricks gray */
        font-weight: bold;
        font-size: 16px;
    }
    .stTextInput > div > input {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #FF3621; /* Databricks orange */
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #E0301E; /* Darker orange on hover */
    }

    /* Results Table */
    .stDataFrame {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        background-color: #FFFFFF;
    }
    .stDataFrame table th {
        background-color: #FF3621;
        color: #FFFFFF;
        font-weight: bold;
    }
    .stDataFrame table td {
        color: #4A4A4A;
    }

    /* Alerts */
    .stAlert {
        border-radius: 5px;
    }

    /* Pie Chart Section */
    .stPlotlyChart, .stImage {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Add a header banner
st.markdown("""
    <div class="header-banner">
        <h1>📄 Resume Screening Assistant for Databricks</h1>
    </div>
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

# Helper functions
def normalize_text(text):
    text = text.lower()
    # Remove underscores, hyphens, and specific phrases, replacing with empty string
    text = re.sub(r'_|-|,\s*collaborated in agile teams|,\s*developed solutions for|,\s*led projects involving|,\s*designed applications with|,\s*built machine learning models for|,\s*implemented data pipelines for|,\s*deployed cloud-based solutions|,\s*optimized workflows for|,\s*contributed to data-driven projects', '', text)
    return text

def check_experience_mismatch(resume, job_description):
    resume_match = re.search(r'(\d+)\s*years?|senior', resume.lower())
    # Simplified pattern to match "X years+" or "senior+"
    job_match = re.search(r'(\d+)\s*years?\+|senior\+', job_description.lower())
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
    # Normalize text for skill matching: replace commas, underscores, and hyphens with spaces
    text_normalized = normalize_text(text)
    text_normalized = re.sub(r'[,_-]', ' ', text_normalized)  # Replace commas, underscores, and hyphens with spaces
    # Check for skills by iterating over skills_list to handle multi-word skills
    found_skill = False
    for skill in skills_list:
        # Escape the skill to handle special characters, and ensure word boundaries
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_normalized):
            found_skill = True
            break
    if is_resume and not found_skill:
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
    bert_tokenized = _bert_tokenizer(bert_inputs, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    t5_inputs = []
    for resume in resumes:
        prompt = re.sub(r'\b[Cc]\+\+\b', 'c++', resume)
        prompt_normalized = normalize_text(prompt)
        t5_inputs.append(f"summarize: {prompt_normalized}")
    t5_tokenized = _t5_tokenizer(t5_inputs, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    return bert_tokenized, t5_inputs, t5_tokenized

@st.cache_data
def extract_skills(text):
    """Extract skills from text in a single pass."""
    text_normalized = normalize_text(text)
    text_normalized = re.sub(r'[,_-]', ' ', text_normalized)
    found_skills = []
    for skill in skills_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_normalized):
            found_skills.append(skill)
    return set(found_skills)

@st.cache_data
def classify_and_summarize_batch(resumes, job_description, _bert_tokenized, _t5_inputs, _t5_tokenized, _job_skills_set):
    bert_tokenizer, bert_model, t5_tokenizer, t5_model, device = st.session_state.models
    bert_tokenized = {k: v.to(device) for k, v in _bert_tokenized.items()}
    
    # BERT inference (batched)
    with torch.no_grad():
        outputs = bert_model(**bert_tokenized)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)
    
    confidence_threshold = 0.85
    results = []
    
    # Batch T5 inference for all resumes
    t5_tokenized = {k: v.to(device) for k, v in _t5_tokenized.items()}
    with torch.no_grad():
        t5_outputs = t5_model.generate(
            t5_tokenized['input_ids'],
            attention_mask=t5_tokenized['attention_mask'],
            max_length=30,
            min_length=8,
            num_beams=4,  # Match notebook
            no_repeat_ngram_size=3,
            length_penalty=3.0,  # Match notebook
            early_stopping=True
        )
    summaries = [t5_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in t5_outputs]
    summaries = [re.sub(r'\s+', ' ', summary).strip() for summary in summaries]
    
    for i, (resume, prob, pred, summary, t5_input) in enumerate(zip(resumes, probabilities, predictions, summaries, _t5_inputs)):
        # Compute skill overlap
        resume_skills_set = extract_skills(resume)
        skill_overlap = len(_job_skills_set.intersection(resume_skills_set)) / len(_job_skills_set) if _job_skills_set else 0

        # Step 1: Check model confidence
        if prob[pred] < confidence_threshold:
            suitability = "Uncertain"
            warning = f"Low confidence: {prob[pred]:.4f}"
        else:
            # Step 2: Check skill irrelevance
            if skill_overlap < 0.4:
                suitability = "Irrelevant"
                warning = "Skills are irrelevant"
            else:
                # Step 3: Determine initial suitability based on skill overlap
                suitability = "Relevant" if skill_overlap >= 0.5 else "Irrelevant"
                warning = "Skills are not a strong match" if suitability == "Irrelevant" else None

                # Step 4: Check experience mismatch and override suitability if necessary
                exp_warning = check_experience_mismatch(resume, job_description)
                if exp_warning:
                    suitability = "Uncertain"
                    warning = exp_warning
        
        # Post-process T5 summary for all resumes (Relevant, Uncertain, or Irrelevant)
        skills = list(set(skills_pattern.findall(t5_input)))  # Deduplicate skills
        exp_match = re.search(r'\d+\s*years?|senior', resume.lower())
        if skills and exp_match:
            summary = f"{', '.join(skills)} proficiency, {exp_match.group(0)} experience"
        else:
            summary = f"{exp_match.group(0) if exp_match else 'unknown'} experience"
        
        results.append({
            "Resume": f"Resume {st.session_state.resumes.index(resume)+1}",
            "Suitability": suitability,
            "Data/Tech Related Skills Summary": summary,
            "Warning": warning or "None"
        })
    
    return results

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
            resume_lower = re.sub(r'[,_-]', ' ', resume_lower)
            found_skills = []
            for skill in skills_list:
                if re.search(rf'\b{re.escape(skill)}\b', resume_lower):
                    found_skills.append(skill)
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
    plt.title("Skill Frequency Across Resumes", fontsize=12, color='#FF3621', pad=10)  # Use Databricks orange for title
    return fig

def main():
    """Main function to run the Streamlit app for resume screening."""
    # Streamlit interface
    with st.sidebar:
        st.markdown("""
            <h1 style='text-align: center; font-size: 32px; margin-bottom: 10px;'>💻 Resume Screening Assistant for Data/Tech</h1>
            <p style='text-align: center; font-size: 16px; margin-top: 0;'>
                Welcome to our AI-powered resume screening tool, specialized for data science and tech roles! This app evaluates multiple resumes against a single job description, providing suitability classifications, skill summaries, and a skill frequency visualization.
            </p>
        """, unsafe_allow_html=True)

        with st.expander("How to Use the App"):
            st.markdown("""
                - Enter up to 5 candidate resumes in the text boxes below, listing data/tech skills and experience (e.g., "Expert in python, databricks, 6 years experience").
                - Enter the job description, specifying required skills and experience (e.g., "Data engineer requires python, spark, 5 years+").
                - Click the "Analyze" button to evaluate all non-empty resumes (at least one resume required).
                - Use the "Add Resume" or "Remove Resume" buttons to adjust the number of resume fields (1-5).
                - Use the "Reset" button to clear all inputs and results.
                - Results can be downloaded as a CSV file for record-keeping.
                - View the skill frequency pie chart to see the distribution of skills across resumes.
            """)

        with st.expander("Example Test Cases"):
            st.markdown("""
                - **Test Case 1**:
                    - Resume 1: "Expert in python, machine learning, tableau, 4 years experience"
                    - Resume 2: "Skilled in sql, pandas, 2 years experience"
                    - Resume 3: "Proficient in java, python, 5 years experience"
                    - Job Description: "Data scientist requires python, machine learning, 3 years+"
                - **Test Case 2**:
                    - Resume 1: "Skilled in databricks, spark, python, 6 years experience"
                    - Resume 2: "Expert in sql, tableau, business intelligence, 3 years experience"
                    - Resume 3: "Proficient in rust, langchain, 2 years experience"
                    - Job Description: "Data engineer requires python, spark, 5 years+"
            """)

        with st.expander("Guidelines"):
            st.markdown("""
                - Use comma-separated skills from a comprehensive list including python, sql, databricks, etc. (79 skills supported, see Project Report for full list).
                - Include experience in years (e.g., "3 years experience" or "1 year experience") or as "senior".
                - Focus on data/tech skills for accurate summarization.
                - Resumes with only irrelevant skills (e.g., sales, marketing) will be classified as "Irrelevant".
            """)

        with st.expander("Classification Criteria"):
            st.markdown("""
                Resumes are classified based on:
                - **Skill Overlap**: The resume's data/tech skills are compared to the job's requirements. A skill overlap below 40% results in an "Irrelevant" classification.
                - **Model Confidence**: A finetuned BERT model evaluates skill relevance. If confidence is below 85%, the classification is "Uncertain".
                - **Experience Match**: The resume's experience (in years or seniority) must meet or exceed the job's requirement.

                **Outcomes**:
                - **Relevant**: Skill overlap ≥ 50%, sufficient experience, and high model confidence (≥ 85%).
                - **Irrelevant**: Skill overlap < 40% or high confidence in low skill relevance.
                - **Uncertain**: Skill overlap ≥ 50% but experience mismatch (e.g., resume has 2 years, job requires 5 years+), or low model confidence (< 85%).

                **Note**: An experience mismatch warning is shown if the resume's experience is below the job's requirement, overriding the skill overlap and confidence to classify as Uncertain.
            """)

    # Initialize session state
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    if 'resumes' not in st.session_state:
        st.session_state.resumes = [""] * 5  # Default to 5 empty resume slots
    if 'num_resumes' not in st.session_state:
        st.session_state.num_resumes = 1  # Start with 1 resume field
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'pie_chart' not in st.session_state:
        st.session_state.pie_chart = None

    # Resume input fields
    st.subheader("Candidate Resumes")
    num_resumes = st.session_state.num_resumes
    for i in range(num_resumes):
        st.session_state.resumes[i] = st.text_input(f"Resume {i+1}", value=st.session_state.resumes[i], key=f"resume_{i}")

    # Buttons to add/remove resume fields
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Add Resume") and num_resumes < 5:
            st.session_state.num_resumes += 1
            st.session_state.results = None  # Clear previous results
            st.session_state.pie_chart = None
            st.rerun()  # Updated to st.rerun()
    with col2:
        if st.button("Remove Resume") and num_resumes > 1:
            st.session_state.num_resumes -= 1
            st.session_state.resumes[num_resumes] = ""  # Clear the removed field
            st.session_state.results = None  # Clear previous results
            st.session_state.pie_chart = None
            st.rerun()  # Updated to st.rerun()
    with col3:
        if st.button("Reset"):
            st.session_state.num_resumes = 1
            st.session_state.resumes = [""] * 5
            st.session_state.job_description = ""
            st.session_state.results = None
            st.session_state.pie_chart = None
            st.rerun()  # Updated to st.rerun()

    # Job description input
    st.subheader("Job Description")
    st.session_state.job_description = st.text_input("Enter the job description (e.g., 'Data engineer requires python, spark, 5 years+')", value=st.session_state.job_description)

    # Analyze button
    if st.button("Analyze"):
        resumes = [resume.strip() for resume in st.session_state.resumes[:num_resumes]]
        job_description = st.session_state.job_description.strip()

        valid_resumes = []
        for i, resume in enumerate(resumes):
            validation_error = validate_input(resume, is_resume=True)
            if validation_error and resume:
                st.error(f"Resume {i+1}: {validation_error}")
            elif resume:
                valid_resumes.append(resume)

        validation_error = validate_input(job_description, is_resume=False)
        if validation_error and job_description:
            st.error(f"Job Description: {validation_error}")

        if valid_resumes and job_description:
            # Extract job skills
            job_skills_set = extract_skills(job_description)

            # Tokenize inputs
            bert_tokenized, t5_inputs, t5_tokenized = tokenize_inputs(valid_resumes, job_description, st.session_state.models[0], st.session_state.models[2])

            # Classify and summarize
            results = classify_and_summarize_batch(valid_resumes, job_description, bert_tokenized, t5_inputs, t5_tokenized, job_skills_set)
            st.session_state.results = results

            # Generate skill frequency pie chart
            pie_chart = generate_skill_pie_chart(valid_resumes)
            st.session_state.pie_chart = pie_chart

    # Display results
    if st.session_state.results:
        st.subheader("Results")
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df, use_container_width=True)

        # Download results as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

    # Display pie chart
    if st.session_state.pie_chart:
        st.subheader("Skill Frequency Across Resumes")
        st.pyplot(st.session_state.pie_chart)
    elif st.session_state.results and not st.session_state.pie_chart:
        st.warning("No recognized data/tech skills found in the resumes to generate a pie chart.")

if __name__ == "__main__":
    main()