import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter
import datetime

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🧠", layout="wide")

# Sidebar: App info and instructions
st.sidebar.title("🧠 Customer Feedback Analyzer")
st.sidebar.markdown("""
Analyze customer feedback for sentiment, emotion, and key themes.
- **Single review** or **batch CSV upload**
- **Sentiment & emotion detection**
- **Theme extraction** (top keywords)
- **Visualizations**: Pie chart, trend line, word cloud
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

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

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
    words = " ".join(texts).lower().split()
    stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0])
    keywords = [w for w in words if w.isalpha() and w not in stopwords]
    return Counter(keywords).most_common(top_n)

# Main UI: Tabs for workflow clarity
tab1, tab2 = st.tabs(["Single Review", "Batch CSV Analysis"])

with tab1:
    st.header("🔍 Analyze a Single Review")
    st.markdown("Paste a customer comment below to see sentiment, emotion, and key themes.")
    text = st.text_area("Enter customer feedback:", help="Paste one review or comment here.")
    if st.button("Analyze Single Review") and text:
        with st.spinner("Analyzing..."):
            sentiment = sentiment_model(text)[0]
            emotion = emotion_model(text)[0]
            st.markdown(
                f"**Sentiment:** <span style='color: {'green' if sentiment['label']=='POSITIVE' else 'red' if sentiment['label']=='NEGATIVE' else 'gray'}'>{get_emoji(sentiment['label'])} {sentiment['label']}</span> (confidence: {sentiment['score']:.2f})",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Emotion:** <span style='color:blue'>{emotion['label']}</span> (confidence: {emotion['score']:.2f})",
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
        date_col = st.selectbox("Select date column (optional for trend)", ["None"] + list(df.columns))
        if st.button("Analyze All Reviews"):
            with st.spinner("Analyzing feedback..."):
                df["Sentiment"] = df[col].astype(str).apply(lambda t: sentiment_model(t)[0]["label"])
                df["SentimentScore"] = df[col].astype(str).apply(lambda t: sentiment_model(t)[0]["score"])
                df["Emotion"] = df[col].astype(str).apply(lambda t: emotion_model(t)[0]["label"])
                df["EmotionScore"] = df[col].astype(str).apply(lambda t: emotion_model(t)[0]["score"])
            st.success("Analysis complete!")
            st.dataframe(df[[col, "Sentiment", "SentimentScore", "Emotion", "EmotionScore"]])

            # Sentiment distribution pie chart
            pie = px.pie(df, names="Sentiment", title="Sentiment Distribution", color="Sentiment",
                         color_discrete_map={"POSITIVE":"green","NEGATIVE":"red","NEUTRAL":"gray"})
            st.plotly_chart(pie, use_container_width=True)

            # Sentiment trend line chart (if date column selected)
            if date_col != "None":
                try:
                    df["Date"] = pd.to_datetime(df[date_col])
                    trend = df.groupby([df["Date"].dt.to_period("M"), "Sentiment"]).size().unstack(fill_value=0)
                    trend.index = trend.index.astype(str)
                    line = px.line(trend, title="Sentiment Trend Over Time")
                    st.plotly_chart(line, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot trend: {e}")

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

# Footer
st.markdown("---")
st.markdown("**About:** This app uses Hugging Face Transformers for sentiment and emotion analysis. For feedback or suggestions, contact the developer.")
