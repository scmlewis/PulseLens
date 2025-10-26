import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(page_title="Customer Feedback Sentiment & Aspect Classifier", page_icon="🧠", initial_sidebar_state="expanded")

# --- Modernized CSS and layout (reference-inspired) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;700&display=swap" rel="stylesheet">
<style>
body, .stApp { background: #181c27 !important; font-family: 'Segoe UI', sans-serif !important; color: #f3f6fb !important; }
.top-banner {
    width: 100vw;
    margin-left: -3.5vw;
    background: linear-gradient(90deg, #4F8BF9 70%, #2D5AAB 100%);
    padding: 1.3rem 0 0.9rem 2.4rem;
    color: #fff;
    box-shadow: 0 2px 12px #0001;
    border-radius: 0 0 1.15rem 1.15rem;
    margin-bottom: 2.1em;
}
.top-banner h1 { font-size: 2.2em; font-weight: 800; margin: 0 0 0.3em 0; color: #fff; }
.top-banner .desc { font-weight: 430; font-size: 1.09em; color: #dbe2f7; }
.card {
    background: #232a3b !important;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(50,60,75,0.12);
    padding: 2rem 2.1rem 1.55rem 2rem;
    margin-bottom: 2.2rem;
}
.stTabs [data-baseweb='tab-list'] { background: none !important; }
.stTabs [data-baseweb='tab'] { font-size: 1.14em; font-weight: 700; color: #a8c7ff !important; background: none;}
.stTabs [data-baseweb='tab'][aria-selected='true'] { color: #fff !important; border-bottom:3px solid #32449b; }
.stButton > button {
    background: linear-gradient(90deg,#32449b,#485cdd);
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.65em 2.2em !important;
    font-size: 1.1em !important;
    margin: 0.18em 0.8em 0.18em 0;
}
.stButton > button:hover {
    background: linear-gradient(90deg,#4658af,#6270f7) !important;
    color: #e7eaff !important;
}
.stTextInput > div>input, .stTextArea textarea, .stSelectbox>div>div {
    background: #22283a !important;
    border-radius: 8px !important;
    color: #f3f6fb !important;
    border: 2px solid #32449b !important;
    font-size: 1.13em !important;
    font-weight: 500 !important;
}
.stTextInput > div>input:focus, .stTextArea textarea:focus, .stSelectbox>div>div:focus {
    border-color: #7e9dde !important;
    box-shadow: 0 0 8px #395fa7 !important;
    outline: none !important;
}
.stDataFrame > div > div { border-radius: 12px !important; box-shadow:0 0 14px #171d2e;}
.st-expander { background: #222945 !important; border-radius: 14px !important; color: #c3cff8;}
/* Sidebar */
[data-testid="stSidebar"] { background: #181c27 !important;}
.sb-section { margin-bottom: 1.3em; }
.sb-hdr { color: #8ebafa !important; font-weight: bold; font-size: 1.04em; margin-bottom:0.18em;}
.sb-list { padding-left:1.15em; margin-top:0.13em; margin-bottom:0.57em;}
.sb-list li { color: #e8eaff; font-size: 1em; margin-bottom:0.15em; line-height: 1.5;}
hr.sb-hr { border: none; border-top: 1px solid #23305a; margin: 0.82em 0; }
</style>
""", unsafe_allow_html=True)

# --- Top Banner outside all cards ---
st.markdown(
    """
    <div class='top-banner'>
        <h1>🧠 Customer Feedback Sentiment & Aspect Classifier</h1>
        <div class='desc'>AI-powered sentiment & aspect mining for feedback across industries</div>
    </div>
    """, unsafe_allow_html=True
)

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()

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
    st.sidebar.markdown("<div class='sb-section'><span class='sb-hdr'>Suggested Aspects</span></div>", unsafe_allow_html=True)
    for group, aspects in GROUPED_ASPECTS.items():
        st.sidebar.markdown(
            f"<div class='sb-section'><span class='sb-hdr'>{group}</span><ul class='sb-list'>" +
            "".join(f"<li>{asp}</li>" for asp in aspects) +
            "</ul></div><hr class='sb-hr'>", unsafe_allow_html=True
        )

render_grouped_aspects()

# --- Welcome Card
st.markdown("""
<div class="card">
    <h3 style="margin-bottom:0.7em;"><span style="font-size:1.2em;margin-right:0.3em;">👋</span>Welcome!</h3>
    <div style="font-size:1.08em; color:#d5daf1">
        Fast, AI-powered feedback analysis. Enter a review (single or batch), select aspects, and classify below.<br>
        <span style='color:#b9bcff;'>Tip: Paste a review list for batch analysis with live charts.</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""<div class="card">""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝 Single Review", "📂 Batch CSV/Manual"])

def sentiment_to_stars(sentiment, score):
    if sentiment == "positive":
        if score > 0.85: return 5
        elif score > 0.7: return 4
        else: return 4
    elif sentiment == "neutral":
        return 3
    else:
        if score > 0.85: return 1
        elif score > 0.7: return 2
        else: return 2

with tab1:
    st.subheader("Single Review")
    def generate_sample(): st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)
    def clear_text(): st.session_state["review_text"] = ""
    col1, col2 = st.columns([1, 1])
    with col1: st.button("✨ Generate Sample", on_click=generate_sample)
    with col2: st.button("🧹 Clear", on_click=clear_text)
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    text = st.text_area("💬 Enter a review", height=120, key="review_text")
    aspects = st.text_input("🔎 Aspects/Categories (comma-separated)", value="", key="aspects_text")
    if st.button("🚦 Classify Now"):
        if not text.strip():
            st.error("Please enter a review.")
        elif not aspects.strip():
            st.error("Please enter at least one aspect.")
        else:
            with st.spinner("Classifying..."):
                aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]
                aspect_result = classifier(text, candidate_labels=aspect_list, multi_label=True)
                sentiment_result = classifier(text, candidate_labels=SENTIMENT_LABELS)
                sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                stars = sentiment_to_stars(sentiment_result['labels'][0], sentiment_result['scores'][0])
                st.markdown(
                    f"<div class='card' style='background: #222945;'>"
                    f"<h4>Sentiment: <span style='color:#8fffa4;'>{sentiment_emoji.get(sentiment_result['labels'][0],'')}</span> <span style='font-size:1.2em;'>{sentiment_result['labels'][0].capitalize()}</span> <span style='color:#b5baff;'>(Score: {sentiment_result['scores'][0]:.2f})</span></h4>"
                    f"<h4>Star Rating: {'⭐'*stars} ({stars}/5)</h4>"
                    f"<h5 style='color:#9bc8ff'>Aspect Relevance Scores:</h5><div>",
                    unsafe_allow_html=True
                )
                df = pd.DataFrame({
                    "Aspect": aspect_result["labels"],
                    "Score": aspect_result["scores"]
                })
                for idx, row in df.iterrows():
                    score = row["Score"]
                    color = "#63b9ff" if score > 0.5 else "#788ba7"
                    emoji = "🔵" if score > 0.7 else "⚪"
                    st.markdown(
                        f"<span style='font-size:1.09em; color:{color};'>{emoji} <b>{row['Aspect']}</b>: {score:.2f}</span>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)

with tab2:
    st.subheader("Batch Reviews (CSV or Manual List)")
    st.markdown(
        "<div style='color:#7e8abb; font-size:1.1em;'><b>Instructions:</b> Upload a UTF-8 CSV with header <code>review</code>, or paste multiple reviews (one per line). Enter aspects comma-separated for analysis below.</div>",
        unsafe_allow_html=True
    )
    with st.expander("Batch Input Options"):
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_file = st.file_uploader("🗂️ CSV file upload", type=["csv"])
        with col2:
            manual_text = st.text_area("📋 Paste reviews (one per line)", height=120)
    aspects = st.text_input("🔎 Aspects/Categories for batch", value="", key="batch_aspects_text")
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
    elif manual_text and manual_text.strip():
        reviews = [line.strip() for line in manual_text.split("\n") if line.strip()]
    if reviews:
        st.write("Sample Reviews", pd.DataFrame({"review": reviews[:5]}))
        if st.button("🚦 Classify Batch"):
            if not aspects.strip():
                st.error("Please enter at least one aspect.")
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
                st.markdown(
                    "<div class='card' style='background: #222945;'><h5 style='color:#9bc8ff'>Batch Classification Results:</h5>",
                    unsafe_allow_html=True
                )
                st.dataframe(results_df)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='card' style='background: #232a3b;'>", unsafe_allow_html=True)
                st.markdown("<h5 style='color:#9bc8ff'>Rating Distribution:</h5>", unsafe_allow_html=True)
                fig1 = px.pie(
                    results_df, names="star_rating", title="Star Rating Distribution", 
                    color_discrete_sequence=px.colors.sequential.Purples, hole=0.3
                )
                fig1.update_traces(textfont_color='white', marker=dict(line=dict(color='#222945', width=2)))
                fig1.update_layout(
                    title_font_color="#bacaff",
                    paper_bgcolor="#232a3b",
                    plot_bgcolor="#232a3b",
                    font_color="#e6eafe"
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("<h5 style='color:#9bc8ff'>Sentiment Analysis:</h5>", unsafe_allow_html=True)
                fig2 = px.bar(
                    results_df, x="sentiment", title="Sentiment Distribution", color="sentiment",
                    color_discrete_map={"positive":"#7ef49e","neutral":"#f7dd8f","negative":"#f47e7e"}
                )
                fig2.update_layout(
                    title_font_color="#bacaff",
                    paper_bgcolor="#232a3b",
                    plot_bgcolor="#232a3b",
                    font_color="#e6eafe"
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_result,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<hr><div style='color:#8aa2ff;font-size:1em;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</div>",
    unsafe_allow_html=True
)
