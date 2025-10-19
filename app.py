import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🧠", layout="wide")

# Sidebar: App info and instructions
st.sidebar.title("🧠 Customer Feedback Analyzer")
st.sidebar.markdown("""
Analyze customer feedback for sentiment, emotion, and business-relevant categories.
- **Single review** or **batch CSV upload**
- **Sentiment, emotion, and category detection**
- **Theme extraction** (top keywords)
- **Visualizations**: Pie chart, word cloud
- **Downloadable results**
""")
st.sidebar.info("Tip: For best results, use feedback in English. For other languages, try multilingual models.")

# Model loading
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

@st.cache_resource
def load_category_model():
    # Example: Replace with a business-topic classifier if available
    # For demonstration, using a generic topic classifier
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()
category_model = load_category_model()

# Define your business categories
CATEGORIES = [
    "Product", "Service", "Staff", "Pricing", "Delivery", "Technical Issues", "Suggestions/Requests", "Other"
]

def get_emoji(label):
    return {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(label, "😐")

def make_wordcloud(texts):
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(texts))
    buf = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

def extract_keywords(texts, top_n=10):
    stopwords = set([
        "the", "and", "is", "in", "it", "of", "to", "a", "for", "on", "with", "this", "that", "was", "are", "as", "but", "be", "at", "by", "an", "or", "from"
    ])
    words = " ".join(texts).lower().split()
    keywords = [w for w in words if w.isalpha() and w not in stopwords]
    return Counter(keywords).most_common(top_n)

def predict_category(text):
    result = category_model(text, candidate_labels=CATEGORIES)
    return result["labels"][0] if result["scores"][0] > 0.4 else "Other"

tab1, tab2 = st.tabs(["Single Review", "Batch CSV Analysis"])

with tab1:
    st.header("🔍 Analyze a Single Review")
    st.markdown("Paste a customer comment below to see sentiment, emotion, and key themes.")
    text = st.text_area("Enter customer feedback:", help="Paste one review or comment here.")
    if st.button("Analyze Single Review") and text:
        with st.spinner("Analyzing..."):
            sentiment = sentiment_model(text)[0]
            emotion = emotion_model(text)[0]
            category = predict_category(text)
            st.markdown(
                f"**Sentiment:** <span style='color: {'green' if sentiment['label']=='POSITIVE' else 'red' if sentiment['label']=='NEGATIVE' else 'gray'}'>{get_emoji(sentiment['label'])} {sentiment['label']}</span> (confidence: {sentiment['score']:.2f})",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Emotion:** <span style='color:blue'>{emotion['label']}</span> (confidence: {emotion['score']:.2f})",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Category:** <span style='color:purple'>{category}</span>",
                unsafe_allow_html=True,
            )
            # Keyword extraction
            keywords = extract_keywords([text])
            st.markdown("**Top Keywords:** " + ", ".join([f"`{kw[0]}`" for kw in keywords]))

with tab2:
    st.header("📊 Batch Feedback Analysis")
    st.markdown("Upload a CSV file with customer feedback. Select the column containing the comments.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Columns detected:", list(df.columns))
        col = st.selectbox("Select feedback column", df.columns)
        if st.button("Analyze All Reviews"):
            with st.spinner("Analyzing feedback..."):
                df["Sentiment"] = df[col].astype(str).apply(lambda t: sentiment_model(t)[0]["label"])
                df["SentimentScore"] = df[col].astype(str).apply(lambda t: sentiment_model(t)[0]["score"])
                df["Emotion"] = df[col].astype(str).apply(lambda t: emotion_model(t)[0]["label"])
                df["EmotionScore"] = df[col].astype(str).apply(lambda t: emotion_model(t)[0]["score"])
                df["Category"] = df[col].astype(str).apply(predict_category)
            st.success("Analysis complete!")
            st.dataframe(df[[col, "Sentiment", "SentimentScore", "Emotion", "EmotionScore", "Category"]])

            # Sentiment distribution pie chart
            pie = px.pie(df, names="Sentiment", title="Sentiment Distribution", color="Sentiment",
                         color_discrete_map={"POSITIVE":"green","NEGATIVE":"red","NEUTRAL":"gray"})
            st.plotly_chart(pie, use_container_width=True)

            # Category distribution
            cat_bar = px.bar(df["Category"].value_counts().reset_index(), x="index", y="Category",
                             labels={"index": "Category", "Category": "Count"}, title="Feedback by Category")
            st.plotly_chart(cat_bar, use_container_width=True)

            # Word cloud
            st.subheader("Word Cloud of Feedback")
            wc_buf = make_wordcloud(df[col].astype(str).tolist())
            st.image(wc_buf)

            # Top keywords
            st.subheader("Top Keywords")
            keywords = extract_keywords(df[col].astype(str).tolist())
            st.markdown(", ".join([f"`{kw[0]}` ({kw[1]})" for kw in keywords]))

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", data=csv, file_name="feedback_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("**About:** This app uses Hugging Face Transformers for sentiment, emotion, and category analysis. For feedback or suggestions, contact the developer.")
