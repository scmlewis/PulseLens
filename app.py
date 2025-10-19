import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Zero-Shot Sentiment & Aspect Classifier", page_icon="🧠", layout="wide")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

# Define common aspects/categories for demonstration
DEFAULT_ASPECTS = [
    "food", "service", "ambience", "price", "delivery", "product quality", "staff", "support", "design", "usability"
]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

st.title("🧠 Zero-Shot Sentiment & Aspect Classifier")
st.markdown("""
This app uses Facebook's BART Large MNLI model for robust, general-purpose sentiment and aspect classification.
- **No model loading errors**
- **Works for any domain**
- **No need for domain-specific ABSA models**
""")

tab1, tab2 = st.tabs(["Single Review", "Batch CSV Reviews"])

with tab1:
    st.subheader("Single Review")
    text = st.text_area("Enter a review:", value="The pasta was delicious, but the service was slow and the ambience was noisy.", height=100)
    aspects = st.text_input("Enter aspects/categories (comma-separated):", value=", ".join(DEFAULT_ASPECTS))
    if st.button("Classify Review"):
        if not text.strip():
            st.info("Please enter a review.")
        else:
            with st.spinner("Classifying..."):
                aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                aspect_result = classifier(text, candidate_labels=aspect_list, multi_label=True)
                sentiment_result = classifier(text, candidate_labels=SENTIMENT_LABELS)
                st.markdown("**Predicted Sentiment:**")
                st.write(f"Label: {sentiment_result['labels'][0]}, Score: {sentiment_result['scores'][0]:.2f}")
                st.markdown("**Aspect Relevance Scores:**")
                df = pd.DataFrame({
                    "Aspect": aspect_result["labels"],
                    "Score": aspect_result["scores"]
                })
                st.dataframe(df)

with tab2:
    st.subheader("Batch Reviews (CSV)")
    csv_file = st.file_uploader("Upload a CSV containing a 'review' column:", type=["csv"])
    aspects = st.text_input("Enter aspects/categories for batch (comma-separated):", value=", ".join(DEFAULT_ASPECTS), key="batch")
    if csv_file:
        dataframe = pd.read_csv(csv_file)
        if 'review' not in dataframe.columns:
            st.warning("Your CSV must contain a column named 'review'.")
        else:
            st.write("Sample Reviews:", dataframe.head())
            if st.button("Classify Uploaded Reviews"):
                aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                results = []
                with st.spinner("Classifying batch reviews..."):
                    for r_idx, r in dataframe['review'].items():
                        sentiment_result = classifier(r, candidate_labels=SENTIMENT_LABELS)
                        aspect_result = classifier(r, candidate_labels=aspect_list, multi_label=True)
                        results.append({
                            "review": r,
                            "sentiment": sentiment_result["labels"][0],
                            "sentiment_score": sentiment_result["scores"][0],
                            "top_aspect": aspect_result["labels"][0],
                            "aspect_score": aspect_result["scores"][0]
                        })
                results_df = pd.DataFrame(results)
                st.markdown("**Batch Classification Results:**")
                st.dataframe(results_df)
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_result,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
st.markdown("---\n*Model: facebook/bart-large-mnli (Meta, Hugging Face)*")
