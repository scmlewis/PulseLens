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

GROUPED_ASPECTS = {
    "🍽️ Restaurant": [
        "food", "service", "ambience", "price", "delivery", "staff", "product quality"
    ],
    "💻 Electronics": [
        "battery", "display", "camera", "performance", "durability", "shipping", "support"
    ],
    "👗 Fashion": [
        "fit", "material", "style", "comfort", "design", "price"
    ],
    "🛒 Supermarket": [
        "freshness", "variety", "checkout", "customer service", "packaging", "speed"
    ],
    "📚 Books": [
        "plot", "characters", "writing", "pacing", "ending", "value"
    ],
    "🏨 Hotel": [
        "cleanliness", "location", "amenities", "room", "wifi", "maintenance"
    ]
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

# --- Sidebar ---
def render_grouped_aspects():
    html_str = "<h2 style='color:#4F8BF9; margin-bottom:4px;'>📝 How to Use</h2>"
    html_str += "<ul style='margin-top:0; padding-left:20px;'>"
    html_str += "<li>Input or generate customer reviews.</li>"
    html_str += "<li>Manually type or paste aspects/categories for analysis.</li>"
    html_str += "<li>Upload CSV or paste multiple reviews for batch analysis.</li></ul><hr>"
    html_str += "<b>Suggested Aspects by Use Case:</b><br>"
    for category, aspects in GROUPED_ASPECTS.items():
        html_str += f"<h3 style='color:#357AB7; margin:6px 0 4px 0;'>{category}</h3>"
        html_str += "<ul style='padding-left: 20px; margin-top:0; margin-bottom:4px;'>"
        for asp in aspects:
            html_str += f"<li>{asp}</li>"
        html_str += "</ul>"
    st.sidebar.markdown(html_str, unsafe_allow_html=True)

render_grouped_aspects()

# --- Main UI ---
st.markdown(
    """
    <div style="background:linear-gradient(90deg, #4F8BF9, #2D5AAB);
    padding: 15px; border-radius: 8px; box-shadow: 2px 2px 10px #888;">
        <h1 style="color: white; margin:0;">🧠 Customer Feedback Sentiment & Aspect Classifier</h1>
        <p style="color: #d0d0d0; font-size:1.1em; margin-top: 8px;">Analyze customer reviews for sentiment and aspect relevance with manual aspect input.</p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_style = """
    <style>
    .stTabs [data-baseweb='tab'] {
        font-size: 1.3em !important;
        padding: 0.5em 1.5em !important;
    }
    </style>
"""
st.markdown(tab_style, unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝 Single Review", "📂 Batch CSV Reviews"])

def sentiment_to_stars(sentiment, score):
    if sentiment == "positive":
        if score > 0.85:
            return 5
        elif score > 0.7:
            return 4
        else:
            return 4
    elif sentiment == "neutral":
        return 3
    else:
        if score > 0.85:
            return 1
        elif score > 0.7:
            return 2
        else:
            return 2

with tab1:
    st.subheader("Single Review")

    def generate_sample():
        st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)

    def clear_text():
        st.session_state["review_text"] = ""

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("✨ Generate Sample Comment", on_click=generate_sample)
    with col2:
        st.button("🧹 Clear", on_click=clear_text)

    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    text = st.text_area("Enter a review:", value=st.session_state["review_text"], height=120, key="review_input")
    aspects = st.text_input("Aspects/Categories (comma-separated):", value="", key="aspects_text")
    if st.button("🔍 Classify Review"):
        if not text.strip():
            st.info("Please enter a review.")
        elif not aspects.strip():
            st.info("Please enter at least one aspect.")
        else:
            with st.spinner("Classifying..."):
                aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                aspect_result = classifier(text, candidate_labels=aspect_list, multi_label=True)
                sentiment_result = classifier(text, candidate_labels=SENTIMENT_LABELS)
                sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                stars = sentiment_to_stars(sentiment_result['labels'][0], sentiment_result['scores'][0])
                st.markdown(
                    f"<h4>Sentiment: <span style='color:green;'>{sentiment_emoji.get(sentiment_result['labels'][0],'')}</span> "
                    f"<span style='font-size:1.2em;'>{sentiment_result['labels'][0].capitalize()}</span> "
                    f"<span style='color:gray;'>(Score: {sentiment_result['scores'][0]:.2f})</span></h4>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<h4>Star Rating: {'⭐'*stars} ({stars}/5)</h4>",
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
    st.subheader("Batch Reviews (CSV or Manual Text)")
    st.markdown(
        "<div style='color:#4F8BF9; font-size:1.1em;'><b>Instructions:</b> Upload a CSV file encoded in <b>UTF-8</b> with a header named <code>review</code> (one review per row), or paste multiple comments below (one per line). Enter aspects comma-separated for analysis.</div>",
        unsafe_allow_html=True
    )
    csv_file = st.file_uploader("Upload a CSV with a 'review' column:", type=["csv"])
    manual_text = st.text_area("Or paste multiple comments here (one per line):", height=120)
    aspects = st.text_input("Aspects/Categories for batch (comma-separated):", value="", key="batch_aspects_text")
    
    reviews = []
    if csv_file:
        try:
            dataframe = pd.read_csv(csv_file, encoding="utf-8")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_file, encoding="latin1")
        if 'review' not in dataframe.columns:
            st.warning("Your CSV must contain a column named 'review'.")
        else:
            reviews = dataframe['review'].dropna().astype(str).tolist()
    elif manual_text.strip():
        reviews = [line.strip() for line in manual_text.split("\n") if line.strip()]
    
    if reviews:
        st.write("Sample Reviews:", pd.DataFrame({"review": reviews[:5]}))
        if st.button("🔍 Classify Batch Reviews"):
            if not aspects.strip():
                st.info("Please enter at least one aspect.")
            else:
                with st.spinner("Classifying batch reviews..."):
                    results = []
                    aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                    for r in reviews:
                        sentiment_result = classifier(r, candidate_labels=SENTIMENT_LABELS)
                        aspect_result = classifier(r, candidate_labels=aspect_list, multi_label=True)
                        stars = sentiment_to_stars(sentiment_result['labels'][0], sentiment_result['scores'][0])
                        results.append({
                            "review": r,
                            "sentiment": sentiment_result["labels"][0],
                            "sentiment_score": sentiment_result["scores"][0],
                            "star_rating": stars,
                            "top_aspect": aspect_result["labels"][0],
                            "aspect_score": aspect_result["scores"][0]
                        })
                results_df = pd.DataFrame(results)
                st.markdown("<h5>Batch Classification Results:</h5>", unsafe_allow_html=True)
                st.dataframe(results_df)
                st.markdown("<h5>Rating Distribution:</h5>", unsafe_allow_html=True)
                fig1 = px.pie(results_df, names="star_rating", title="Star Rating Distribution", color_discrete_sequence=px.colors.sequential.Blues)
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("<h5>Sentiment Analysis:</h5>", unsafe_allow_html=True)
                fig2 = px.bar(results_df, x="sentiment", title="Sentiment Distribution", color="sentiment", color_discrete_map={"positive":"green","neutral":"gray","negative":"red"})
                st.plotly_chart(fig2, use_container_width=True)
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_result,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

st.markdown("<hr><span style='color:gray;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</span>", unsafe_allow_html=True)
