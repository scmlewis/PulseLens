import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(page_title="Customer Feedback Sentiment & Aspect Classifier", page_icon="🧠", layout="wide")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

# Grouped Suggested Aspects by Use Case
GROUPED_ASPECTS = {
    "🍽️ Restaurant": ["food", "service", "ambience", "price", "delivery", "staff", "product quality"],
    "💻 Electronics": ["battery", "display", "camera", "performance", "durability", "shipping", "support"],
    "👗 Fashion": ["fit", "material", "style", "comfort", "design", "price"],
    "🛒 Supermarket": ["freshness", "variety", "checkout", "customer service", "packaging", "speed"],
    "📚 Books": ["plot", "characters", "writing", "pacing", "ending", "value"],
    "🏨 Hotel": ["cleanliness", "location", "amenities", "room", "wifi", "maintenance"]
}

SENTIMENT_LABELS = ["positive", "neutral", "negative"]

SAMPLE_COMMENTS = [
    "I visited the restaurant last night and was impressed by the cozy ambience and friendly staff. The food was delicious, especially the pasta, but the wait time for our main course was a bit long. Overall, a pleasant experience and I would recommend it to friends.",
    "This smartphone has a stunning display and the battery lasts all day, even with heavy use. However, the camera struggles in low light and the device sometimes gets warm during gaming sessions. Customer support was helpful when I had questions about the warranty.",
    "The dress I ordered online arrived quickly and the material feels premium. The fit is true to size and the color matches the photos perfectly. I received several compliments at the event, but I wish the price was a bit lower.",
    "Shopping at this supermarket is always convenient. The produce section is well-stocked and the staff are courteous. However, the checkout lines can get long during weekends and some items are more expensive compared to other stores.",
    "This novel captivated me from the first page. The plot twists kept me guessing, and the characters were well-developed. The pacing slowed down in the middle, but the ending was satisfying. Highly recommended for fans of mystery and drama.",
    "Our stay at the hotel was comfortable. The room was clean and spacious, and the staff were attentive to our needs. The breakfast buffet had a good variety, but the Wi-Fi connection was unreliable at times. The location is perfect for sightseeing."
]

def render_grouped_aspects():
    html_str = """
Analyze customer reviews for sentiment and aspect relevance with manual aspect input.

`review` (one review per row), or paste multiple comments below (one per line). Enter aspects comma-separated for analysis.
"""
    st.sidebar.markdown(html_str)

# --- Initialize session state for user input ---
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# --- Sidebar ---
st.sidebar.title("Settings")
render_grouped_aspects()

st.sidebar.markdown("## Select Use Case Aspect Group")
use_case = st.sidebar.selectbox("Choose a use case:", list(GROUPED_ASPECTS.keys()))
suggested_aspects = GROUPED_ASPECTS[use_case]
default_aspects = ", ".join(suggested_aspects)
aspect_input = st.sidebar.text_input("Aspects to analyze (comma separated):", value=default_aspects)

# --- Main content ---

st.title("Customer Feedback Sentiment & Aspect Classifier")

# Buttons with callbacks to update session state
def generate_sample():
    st.session_state.user_text = random.choice(SAMPLE_COMMENTS)

def clear_text():
    st.session_state.user_text = ""

col_gen, col_clear = st.columns([1, 1])
with col_gen:
    st.button("🎲 Generate Sample Comment", on_click=generate_sample)
with col_clear:
    st.button("🧹 Clear", on_click=clear_text)

# Text area linked to session state so it updates dynamically
user_input = st.text_area("Enter customer feedback:", value=st.session_state.user_text, height=180)

# Split input into lines if multiple comments
comments = [line.strip() for line in user_input.split("\n") if line.strip()]

if comments:
    results = []

    for comment in comments:
        # Run zero-shot classification for aspects
        aspect_result = classifier(comment, candidate_labels=[a.strip() for a in aspect_input.split(",")], multi_label=True)
        # Run zero-shot classification for sentiment
        sentiment_result = classifier(comment, candidate_labels=SENTIMENT_LABELS)

        aspect_scores = dict(zip(aspect_result["labels"], aspect_result["scores"]))
        sentiment = sentiment_result["labels"][0]
        sentiment_score = sentiment_result["scores"][0]

        results.append({
            "comment": comment,
            "predicted_sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "aspect_scores": aspect_scores
        })

    # Display results as a dataframe
    df = pd.DataFrame(results)
    # Explode aspect_scores dictionary into separate columns
    aspect_df = df["aspect_scores"].apply(pd.Series)
    final_df = pd.concat([df.drop(columns=["aspect_scores"]), aspect_df], axis=1)

    st.markdown("### Analysis Results")
    st.dataframe(final_df)

    # Example plot: Sentiment distribution
    fig = px.histogram(final_df, x="predicted_sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter customer feedback to analyze sentiment and aspects.")
