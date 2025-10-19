import streamlit as st
import pandas as pd
import random
from transformers import pipeline

st.set_page_config(page_title="Customer Feedback Sentiment & Aspect Classifier", page_icon="🧠", layout="wide")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

# Expanded aspect suggestions from multiple domains
SUGGESTED_ASPECTS = [
    "food", "service", "ambience", "price", "delivery", "product quality", "staff", "support", "design", "usability",
    "battery", "display", "camera", "performance", "durability", "shipping", "fit", "material", "style", "comfort",
    "cleanliness", "location", "amenities", "checkout", "variety", "freshness", "customer service", "packaging", "speed",
    "plot", "characters", "writing", "pacing", "ending", "value", "features", "sound", "wifi", "room", "maintenance"
]
DEFAULT_ASPECTS = [
    "food", "service", "ambience", "price", "delivery", "product quality", "staff", "support", "design", "usability"
]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

SAMPLE_COMMENTS = [
    "I visited the restaurant last night and was impressed by the cozy ambience and friendly staff. The food was delicious, especially the pasta, but the wait time for our main course was a bit long. Overall, a pleasant experience and I would recommend it to friends.",
    "This smartphone has a stunning display and the battery lasts all day, even with heavy use. However, the camera struggles in low light and the device sometimes gets warm during gaming sessions. Customer support was helpful when I had questions about the warranty.",
    "The dress I ordered online arrived quickly and the material feels premium. The fit is true to size and the color matches the photos perfectly. I received several compliments at the event, but I wish the price was a bit lower.",
    "Shopping at this supermarket is always convenient. The produce section is well-stocked and the staff are courteous. However, the checkout lines can get long during weekends and some items are more expensive compared to other stores.",
    "This novel captivated me from the first page. The plot twists kept me guessing, and the characters were well-developed. The pacing slowed down in the middle, but the ending was satisfying. Highly recommended for fans of mystery and drama.",
    "Our stay at the hotel was comfortable. The room was clean and spacious, and the staff were attentive to our needs. The breakfast buffet had a good variety, but the Wi-Fi connection was unreliable at times. The location is perfect for sightseeing."
]

# --- Sidebar with clickable aspect suggestions ---
st.sidebar.markdown(
    "<h2 style='color:#4F8BF9;'>📝 How to Use</h2>"
    "<ul>"
    "<li>Enter or generate a customer review in the main area.</li>"
    "<li>Click suggested aspects to add them to the input below.</li>"
    "<li>Click <b>Classify Review</b> to analyze sentiment and aspect relevance.</li>"
    "<li>Upload a CSV with a <b>review</b> column for batch analysis.</li>"
    "</ul>"
    "<hr>"
    "<b>Suggested Aspects (click to add):</b>",
    unsafe_allow_html=True
)

# Aspect input state
if "aspects_input" not in st.session_state:
    st.session_state["aspects_input"] = ", ".join(DEFAULT_ASPECTS)

# Clickable aspect buttons
aspect_cols = st.sidebar.columns(3)
for idx, aspect in enumerate(SUGGESTED_ASPECTS):
    if aspect_cols[idx % 3].button(aspect, key=f"aspect_{aspect}"):
        current = st.session_state["aspects_input"]
        aspects_set = set([a.strip() for a in current.split(",") if a.strip()])
        aspects_set.add(aspect)
        st.session_state["aspects_input"] = ", ".join(sorted(aspects_set))

st.sidebar.markdown(
    "<hr><b>Tip:</b> Click any aspect above to add it to your analysis.",
    unsafe_allow_html=True
)

# --- Main UI ---
st.markdown(
    "<h1 style='color:#4F8BF9; font-size:2.5em;'>🧠 Customer Feedback Sentiment & Aspect Classifier</h1>"
    "<hr style='border:1px solid #4F8BF9;'>"
    "<span style='font-size:1.2em;'>Analyze customer reviews for sentiment and aspect relevance using a robust, general-purpose AI model.</span>",
    unsafe_allow_html=True
)

# --- Custom tab buttons with larger font and emoji ---
tab_style = """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.3em !important;
        padding: 0.5em 1.5em !important;
    }
    </style>
"""
st.markdown(tab_style, unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝 Single Review", "📂 Batch CSV Reviews"])

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
    aspects = st.text_input("Aspects/Categories (comma-separated):", value=st.session_state["aspects_input"], key="aspects_input_box")
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
                for idx, row in df.iterrows():
                    score = row["Score"]
                    color = "#4F8BF9" if score > 0.5 else "#aaa"
                    emoji = "🔵" if score > 0.7 else "⚪"
                    st.markdown(
                        f"<span style='font-size:1.1em; color:{color};'>{emoji} <b>{row['Aspect']}</b>: {score:.2f}</span>",
                        unsafe_allow_html=True
                    )

with tab2:
    st.subheader("Batch Reviews (CSV)")
    st.markdown(
        "<div style='color:#4F8BF9; font-size:1.1em;'><b>Instructions:</b> Upload a CSV file encoded in <b>UTF-8</b> with a header named <code>review</code>. Each row should contain a single review text.</div>",
        unsafe_allow_html=True
    )
    csv_file = st.file_uploader("Upload a CSV with a 'review' column:", type=["csv"])
    aspects = st.text_input("Aspects/Categories for batch (comma-separated):", value=st.session_state["aspects_input"], key="batch")
    if csv_file:
        try:
            dataframe = pd.read_csv(csv_file, encoding="utf-8")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_file, encoding="latin1")
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
                st.dataframe(results_df)
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_result,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
                # Table remains visible after download

st.markdown("<hr><span style='color:gray;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</span>", unsafe_allow_html=True)
