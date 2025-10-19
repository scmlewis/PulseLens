import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

st.title("🧠 Customer Feedback Analyzer")
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    - Analyze single feedback or upload a CSV file.
    - For CSV, select the column with feedback text.
    - View sentiment distribution and word cloud.
    - Download results for further analysis.
    """
)

with st.spinner("Loading sentiment model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        st.stop()

input_mode = st.radio("Select input type:", ["Single review", "Upload CSV"])

if input_mode == "Single review":
    text = st.text_area("Enter customer feedback:")
    if st.button("Analyze") and text:
        result = model(text)[0]
        st.markdown(
            f"**Sentiment:** <span style='color: {'green' if result['label']=='POSITIVE' else 'red' if result['label']=='NEGATIVE' else 'gray'}'>{get_emoji(result['label'])} {result['label']}</span> (confidence: {result['score']:.2f})",
            unsafe_allow_html=True,
        )
else:
    file = st.file_uploader("Upload CSV with feedback column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Columns detected:", list(df.columns))
        col = st.selectbox("Select feedback column", df.columns)
        if st.button("Analyze All"):
            with st.spinner("Analyzing feedback..."):
                df["Sentiment"] = df[col].astype(str).apply(lambda t: model(t)[0]["label"])
                df["Confidence"] = df[col].astype(str).apply(lambda t: model(t)[0]["score"])
            st.success("Analysis complete!")
            st.dataframe(df[[col, "Sentiment", "Confidence"]])

            # Sentiment distribution
            fig = px.histogram(df, x="Sentiment", color="Sentiment", title="Sentiment Distribution", color_discrete_map={"POSITIVE":"green","NEGATIVE":"red","NEUTRAL":"gray"})
            st.plotly_chart(fig)

            # Word cloud
            st.subheader("Word Cloud of Feedback")
            wc_buf = make_wordcloud(df[col].astype(str).tolist())
            st.image(wc_buf)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", data=csv, file_name="feedback_results.csv", mime="text/csv")
