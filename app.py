import streamlit as st
import pandas as pd
import random
from transformers import pipeline

st.set_page_config(page_title="Customer Feedback Sentiment & Aspect Classifier", page_icon="🧠", layout="wide")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

DEFAULT_ASPECTS = [
    "food", "service", "ambience", "price", "delivery", "product quality", "staff", "support", "design", "usability"
]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

SAMPLE_COMMENTS = [
    # Restaurant
    "I visited the restaurant last night and was impressed by the cozy ambience and friendly staff. The food was delicious, especially the pasta, but the wait time for our main course was a bit long. Overall, a pleasant experience and I would recommend it to friends.",
    # Electronics
    "This smartphone has a stunning display and the battery lasts all day, even with heavy use. However, the camera struggles in low light and the device sometimes gets warm during gaming sessions. Customer support was helpful when I had questions about the warranty.",
    # Fashion
    "The dress I ordered online arrived quickly and the material feels premium. The fit is true to size and the color matches the photos perfectly. I received several compliments at the event, but I wish the price was a bit lower.",
    # Supermarket
    "Shopping at this supermarket is always convenient. The produce section is well-stocked and the staff are courteous. However, the checkout lines can get long during weekends and some items are more expensive compared to other stores.",
    # Books
    "This novel captivated me from the first page. The plot twists kept me guessing, and the characters were well-developed. The pacing slowed down in the middle, but the ending was satisfying. Highly recommended for fans of mystery and drama.",
    # Hotel
    "Our stay at the hotel was comfortable. The room was clean and spacious, and the staff were attentive to our needs. The breakfast buffet had a good variety, but the Wi-Fi connection was unreliable at times. The location is perfect for sightseeing."
]

st.sidebar.markdown(
    "<h2 style='color:#4F8BF9;'>📝 How to Use</h2>"
    "<ul>"
    "<li>Enter or generate a customer review in the main area.</li>"
    "<li>Optionally, edit the list of aspects/categories.</li>"
    "<li>Click <b>Classify Review</b> to analyze sentiment and aspect relevance.</li>"
    "<li>Upload a CSV with a <b>review</b> column for batch analysis.</li>"
    "</ul>"
    "<hr>"
    "<b>Suggested Aspects:</b><br>"
    "<span style='color:#4F8BF9;'>food, service, ambience, price, delivery, product quality, staff, support, design, usability</span>",
    unsafe_allow_html=True
)

st.header("🧠 Customer Feedback Sentiment & Aspect Classifier", divider="blue")
st.markdown(
    "<span style='font-size:1.2em;'>Analyze customer reviews for sentiment and aspect relevance using a robust, general-purpose AI model.</span>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Single Review", "Batch CSV Reviews"])

with tab1:
    st.subheader("Single Review")
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✨ Generate Sample Comment"):
            st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)
    with col2:
        if st.button("🧹 Clear"):
            st.session_state["review_text"] = ""
    text = st.text_area("Enter a review:", value=st.session_state["review_text"], height=120, key="review_input")
    aspects = st.text_input("Aspects/Categories (comma-separated):", value=", ".join(DEFAULT_ASPECTS))
    if st.button("🔍 Classify Review"):
        if not text.strip():
            st.info("Please enter a review.")
        else:
            with st.spinner("Classifying..."):
                aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                aspect_result = classifier(text, candidate_labels=aspect_list, multi_label=True)
                sentiment_result = classifier(text, candidate_labels=SENTIMENT_LABELS)
                sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                st.markdown(
                    f"<h4>Sentiment: <span style='color:green;'>{sentiment_emoji.get(sentiment_result['labels'][0],'')}</span> <span style='font-size:1.2em;'>{sentiment_result['labels'][0].capitalize()}</span> <span style='color:gray;'>(Score: {sentiment_result['scores'][0]:.2f})</span></h4>",
                    unsafe_allow_html=True
                )
                st.markdown("<h5>Aspect Relevance Scores:</h5>", unsafe_allow_html=True)
                df = pd.DataFrame({
                    "Aspect": aspect_result["labels"],
                    "Score": aspect_result["scores"]
                })
                st.dataframe(df.style.background_gradient(cmap="Blues"))

with tab2:
    st.subheader("Batch Reviews (CSV)")
    csv_file = st.file_uploader("Upload a CSV with a 'review' column:", type=["csv"])
    aspects = st.text_input("Aspects/Categories for batch (comma-separated):", value=", ".join(DEFAULT_ASPECTS), key="batch")
    if csv_file:
        dataframe = pd.read_csv(csv_file)
        if 'review' not in dataframe.columns:
            st.warning("Your CSV must contain a column named 'review'.")
        else:
            st.write("Sample Reviews:", dataframe.head())
            if st.button("🔍 Classify Uploaded Reviews"):
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
                st.markdown("<h5>Batch Classification Results:</h5>", unsafe_allow_html=True)
                st.dataframe(results_df.style.background_gradient(cmap="Blues"))
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_result,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
st.markdown("<hr><span style='color:gray;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</span>", unsafe_allow_html=True)
