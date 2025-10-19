import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Domain-Aware Aspect Sentiment Analyzer", page_icon="🧠", layout="wide")

DOMAIN_MODELS = {
    "Restaurant": "SayakPaul/Restaurant-ABSA-BART",  # aspects: food, service, ambience, drinks
    "Electronics": "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-electronics-aspect",  # aspects: battery, display, price, support, shipping
    "Fashion": "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-fashion-aspect",      # aspects: size, fit, material, delivery, style
    "Hotel": "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-hotel-aspect",          # aspects: room, staff, location, cleanliness, amenities
    "Supermarket": "jordiclive/FABSA",   # aspects: product variety, freshness, staff, pricing, checkout, organization
    "Books": "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-books-aspect",    # aspects: plot, character, pacing, writing, price
    # Add more domains/models as needed
}

st.sidebar.header("Domain & Model Selection")
domain = st.sidebar.selectbox(
    "Select the context (domain) of feedback:",
    options=list(DOMAIN_MODELS.keys()),
    help="Choose the type of product/service being reviewed"
)
model_name = DOMAIN_MODELS[domain]
st.sidebar.markdown(f"**Model:** `{model_name}`")

@st.cache_resource
def load_absa_model(model_name):
    return pipeline("aspect-based-sentiment-analysis", model=model_name)

try:
    absa = load_absa_model(model_name)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"Error loading model: {e}")

st.title("🧠 Aspect-Based Sentiment Analyzer")
st.markdown("""
Analyze reviews for various domains (restaurant, electronics, fashion, hotels, supermarket, books, etc).
Extract aspects, sentiment polarity, opinions, and aspect categories to understand feedback trends!
""")

tab1, tab2 = st.tabs(["Single Review", "Batch CSV Reviews"])

with tab1:
    st.subheader(f"Single Review - Context: {domain}")
    example_texts = {
        "Restaurant": "The pasta was delicious, but the service was slow and the ambience was noisy.",
        "Electronics": "Battery life is impressive but the display is too dim in sunlight.",
        "Fashion": "Dress material is fantastic, but sizing runs small.",
        "Hotel": "Lovely staff and clean rooms, location a bit far from the city center.",
        "Supermarket": "Great variety of fresh fruits, but checkout lines are too long.",
        "Books": "The plot was engaging but character development was weak."
    }
    text = st.text_area("Enter a review:", value=example_texts.get(domain, ""), height=100)
    if st.button("Analyze Review"):
        if not model_loaded:
            st.error("Model could not be loaded. Please try another domain or check Hugging Face.")
        elif not text.strip():
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
    st.subheader(f"Batch Reviews (CSV) - Context: {domain}")
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
st.markdown("---\n*Models powered by Hugging Face Transformers. Choose domain for context-aware analysis!*")
