import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Aspect-Based Sentiment Analyzer", page_icon="🧠", layout="wide")

@st.cache_resource
def load_absa_model():
    return pipeline("aspect-based-sentiment-analysis", model="SayakPaul/Restaurant-ABSA-BART")

absa = load_absa_model()

st.title("🧠 Aspect-Based Sentiment Analyzer")
st.markdown("""
Analyze reviews for aspects, opinions, sentiment polarity, and categories.
This app uses a robust, public ABSA model suitable for most English feedback.
""")

tab1, tab2 = st.tabs(["Single Review", "Batch CSV Reviews"])

with tab1:
    st.subheader("Single Review")
    example = "The pasta was delicious, but the service was slow and the ambience was noisy."
    text = st.text_area("Enter a review:", value=example, height=100)
    if st.button("Analyze Review"):
        if not text.strip():
            st.info("Please enter a review.")
        else:
            with st.spinner("Extracting triplets..."):
                triplets = absa(text)
                if triplets and isinstance(triplets, list):
                    df = pd.DataFrame(triplets)
                    st.markdown("**Predicted Aspect-Opinion-Polarity-Category Triplets:**")
                    st.dataframe(df)
                else:
                    st.info("No aspects/opinions could be extracted from this review.")

with tab2:
    st.subheader("Batch Reviews (CSV)")
    csv_file = st.file_uploader("Upload a CSV containing a 'review' column:", type=["csv"])
    if csv_file:
        dataframe = pd.read_csv(csv_file)
        if 'review' not in dataframe.columns:
            st.warning("Your CSV must contain a column named 'review'.")
        else:
            st.write("Sample Reviews:", dataframe.head())
            if st.button("Analyze Uploaded Reviews"):
                triplet_rows = []
                with st.spinner("Analyzing batch reviews..."):
                    for r_idx, r in dataframe['review'].items():
                        extracted = absa(r)
                        if extracted and isinstance(extracted, list):
                            for triplet in extracted:
                                triplet_rows.append({"review": r, **triplet})
                if triplet_rows:
                    results_df = pd.DataFrame(triplet_rows)
                    st.markdown("**All Extracted Aspect-Opinion-Polarity-Category Triplets:**")
                    st.dataframe(results_df)
                    # Download button for results
                    csv_result = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Triplet Results as CSV",
                        data=csv_result,
                        file_name="triplet_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No triplets found in uploaded reviews.")
st.markdown("---\n*Model: SayakPaul/Restaurant-ABSA-BART (Hugging Face)*")
