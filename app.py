import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(page_title="Customer Feedback Sentiment & Aspect Classifier", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---- Modern CSS for wide layout, buttons, tabs, cards, sidebar grid ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
<style>
.stApp { background: #181c27 !important; font-family: 'Inter', sans-serif !important; color: #f3f6fb !important;}
.main-container {max-width: 1180px; margin: auto;}
.header-banner {
    background: linear-gradient(90deg,#4F8BF9 65%, #2D5AAB 100%);
    border-radius:1.15rem; padding:1.17em 2.3em 1.13em 2.3em; box-shadow:0 2px 16px #23274629;
    text-align:left; border-bottom: 4px solid #43a2ff; margin-bottom: 2em;
}
.header-banner h1 { font-size: 2.07em; font-weight: 900; margin: 0 0 0.14em 0; color: #fff; line-height:1.1; }
.header-banner .desc { font-weight: 500; font-size: 1.12em; color: #e2ebff; margin-top:0.19em; }
.stTabs [data-baseweb='tab-list'] { background: none !important; justify-content:center;}
.stTabs [data-baseweb='tab'] {
    border-radius: 22px 22px 0 0 !important;
    margin-right: 1.2em !important;
    font-size: 1.18em !important; font-weight: 800 !important; color: #adc8ff !important;
    padding: 0.89em 2.33em 0.85em 2.33em !important;
    background: #232a3b !important; box-shadow: 0 2.5px 16px #22283a35;
    border: 2.1px solid #3F59B844;
}
.stTabs [data-baseweb='tab'][aria-selected='true'] {
    color: #fff !important; background: linear-gradient(90deg, #527afe, #3553c3 90%);
    border-bottom: 3.1px solid #8dc7fc !important; box-shadow: 0 2px 12px #325aee18;
}
.stButton > button {
    background: linear-gradient(90deg,#32449b,#485cdd);
    color: #fff !important; font-weight: 700 !important; border-radius: 8px !important;
    padding: 0.68em 2.2em; font-size: 1.09em !important;border:none;box-shadow:0 1px 9px #18113319;}
.stButton>button:hover, .stButton>button:focus {
    background: linear-gradient(90deg,#3970e8,#61a3fd) !important; color: #fff !important;}
.stTextInput > div>input, .stTextArea textarea, .stSelectbox>div>div {
    background: #22283a !important; border-radius: 8px !important; color: #e3f2ff !important;
    border: 2px solid #3a4ece !important; font-size: 1.10em !important; font-weight: 500 !important;}
.stTextInput > div>input:focus, .stTextArea textarea:focus, .stSelectbox>div>div:focus {
    border-color: #61a3fd !important; box-shadow: 0 0 8px #395fa7 !important;}
.card { background: #232a3b; border-radius: 13px;
    box-shadow: 0 5px 20px #1c223510;
    padding: 1.5rem 2.1rem 1.43rem 2.1rem; margin-bottom: 1.5rem;}
.stDataFrame >div>div { border-radius: 11px !important; box-shadow:0 0 14px #151d2e;}
.st-expander { background: #22304a !important; border-radius: 14px !important; color: #e3eefd;}
[data-testid="stSidebar"] { background: #181c27 !important; min-width:300px; width:340px;}
/* Sidebar chips in expandable grid */
.sidebar-aspect-grid { display: flex; flex-wrap: wrap; gap:1em 0.7em; margin-bottom: 1.2em;}
.sidebar-aspect-group { min-width: 140px; max-width: 210px; background:#232a3b; border-radius:10px;
     padding:0.61em 0.77em 0.66em 0.77em; margin-bottom:0.2em;}
.aspect-group-title { font-size:1.01em;color:#7fb6ff;font-weight:700; margin-bottom:0.13em; }
.aspect-chip {
    display:inline-block;background:#293053;color:#f2f4ff;border-radius: 7px;
    padding: 0.21em 0.71em;margin:0.12em 0.20em 0.11em 0;font-size:0.93em;letter-spacing:0.02em;}
/* Output card */
.output-card {
    background: #232a3b; border-radius:11px; box-shadow: 0 2px 12px #191c2e35; 
    padding: 1.18em 2.05em 1.12em 2.07em; margin-bottom:1.27em;
    color: #f3f6fb; font-size:1.15em; letter-spacing:0.006em;
}
.output-stars { margin:0.16em 0 0.29em 0; font-size:1.31em;}
.senti-positive { color:#90ffb8;font-weight:700;}
.senti-label { font-weight:900; font-size:1.13em; margin-right:0.24em;}
.senti-score { color:#b5b9fc; font-size:0.98em; margin-left:0.12em;}
.aspect-row { display:flex;align-items:center;margin:0.16em 0;}
.aspect-dot {
    width:15px;height:15px;display:inline-block;border-radius:24px;
    background:linear-gradient(145deg,#5ae0fb,#1e61e6 80%);
    margin-right: 0.24em;}
.aspect-dot-lo {background:linear-gradient(145deg,#adc6ff,#434766 80%);}
.aspect-score {font-size:1.03em;font-weight:500;margin-left:0.27em;}
</style>
""", unsafe_allow_html=True)

# --- Wide and centered main container ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Header ---
st.markdown(
    """
    <div class="header-banner">
        <h1>🧠 Customer Feedback Sentiment & Aspect Classifier</h1>
        <div class="desc">Modern, AI-powered feedback analytics for all industries.</div>
    </div>
    """,
    unsafe_allow_html=True
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

# --- Sidebar: Help at top, then expandable aspects grid ---
st.sidebar.markdown(
    """
    <div style='background: #202c45; border-left: 4px solid #4F8BF9; border-radius: 10px; padding: 1em 1.3em 1em 1em;
    margin-bottom: 1.1em; color: #bae2ff; font-size: 1.09em; line-height: 1.7;'>
        <span style="font-size:1.13em; margin-right:0.33em;">💡</span>
        Enter a review, choose aspects (comma-separated), then classify. For batch analysis, upload a CSV or paste reviews below. Results will show live sentiment, aspect rating, and charts. All results are processed securely and instantly in your browser.
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar.expander("Suggested Aspects", expanded=False):
    grid_html = '<div class="sidebar-aspect-grid">'
    for group, aspects in GROUPED_ASPECTS.items():
        grid_html += '<div class="sidebar-aspect-group">'
        grid_html += f'<div class="aspect-group-title">{group}</div>'
        for asp in aspects:
            grid_html += f'<span class="aspect-chip">{asp}</span>'
        grid_html += '</div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

# ---- Main tabs (2 large, rounded, styled) ----
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SAMPLE_COMMENTS = [
    "I visited the restaurant last night and was impressed by the cozy ambience and friendly staff. The food was delicious, especially the pasta, but the wait time for our main course was a bit long. Overall, a pleasant experience and I would recommend it to friends.",
    "This smartphone has a stunning display and the battery lasts all day, even with heavy use. However, the camera struggles in low light and the device sometimes gets warm during gaming sessions. Customer support was helpful when I had questions about the warranty.",
    "The dress I ordered online arrived quickly and the material feels premium. The fit is true to size and the color matches the photos perfectly. I received several compliments at the event, but I wish the price was a bit lower.",
    "Shopping at this supermarket is always convenient. The produce section is well-stocked and the staff are courteous. However, the checkout lines can get long during weekends and some items are more expensive compared to other stores.",
    "This novel captivated me from the first page. The plot twists kept me guessing, and the characters were well-developed. The pacing slowed down in the middle, but the ending was satisfying. Highly recommended for fans of mystery and drama.",
    "Our stay at the hotel was comfortable. The room was clean and spacious, and the staff were attentive to our needs. The breakfast buffet had a good variety, but the Wi-Fi connection was unreliable at times. The location is perfect for sightseeing."
]

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

tab1, tab2 = st.tabs(["💬 Single Review", "📊 Batch Reviews"])
with tab1:
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
                    f'''<div class="output-card">
                      <span class="senti-label">Sentiment:</span> <span class="senti-positive">{sentiment_emoji.get(sentiment_result['labels'][0],'')}</span> <b class="senti-positive">{sentiment_result['labels'][0].capitalize()}</b>
                      <span class="senti-score">(Score: {sentiment_result['scores'][0]:.2f})</span>
                      <div class="output-stars">Star Rating: {'⭐'*stars} ({stars}/5)</div>
                      <div style="font-size:1.05em;color:#a7c3fe;font-weight:700;margin:0.23em 0 0.35em 0;">Aspect Relevance Scores:</div>''',
                    unsafe_allow_html=True
                )
                df = pd.DataFrame({
                    "Aspect": aspect_result["labels"],
                    "Score": aspect_result["scores"]
                })
                for idx, row in df.iterrows():
                    colorclass = "aspect-dot" if row["Score"] > 0.6 else "aspect-dot aspect-dot-lo"
                    st.markdown(
                        f'''<div class="aspect-row">
                            <span class="{colorclass}"></span>
                            <span style="font-weight:700;color:#7ecefa;">{row['Aspect']}</span>
                            <span class="aspect-score">: {row["Score"]:.2f}</span>
                        </div>''', unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div style='background: #22304a;border-radius: 11px;padding:0.7em 1.12em 0.81em 1.18em;color:#e3f1fe;font-size:1.06em;
          margin-bottom:1.05em;'><b>Instructions:</b> Upload a UTF-8 CSV file with a column <code>review</code> or paste reviews (one per line) below.<br>
          Enter aspects and press 'Classify Batch' to view results and charts.
        </div>
        """, unsafe_allow_html=True)
    with st.expander("Batch Input Options"):
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_file = st.file_uploader("🗂️ CSV Upload", type=["csv"])
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
                    '''<div class="output-card"><h5 style="color:#a7c3fe;font-size:1.11em;">Batch Classification Results:</h5>''',
                    unsafe_allow_html=True
                )
                st.dataframe(results_df)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('''<div class="card" style="background: #232a3b;">''', unsafe_allow_html=True)
                st.markdown("<h5 style='color:#9bc8ff'>Rating Distribution:</h5>", unsafe_allow_html=True)
                fig1 = px.pie(results_df, names="star_rating", title="Star Rating Distribution", color_discrete_sequence=px.colors.sequential.Purples, hole=0.3)
                fig1.update_traces(textfont_color='white', marker=dict(line=dict(color='#222945', width=2)))
                fig1.update_layout(title_font_color="#bacaff", paper_bgcolor="#232a3b", plot_bgcolor="#232a3b", font_color="#e6eafe")
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("<h5 style='color:#9bc8ff'>Sentiment Analysis:</h5>", unsafe_allow_html=True)
                fig2 = px.bar(
                    results_df, x="sentiment", title="Sentiment Distribution", color="sentiment",
                    color_discrete_map={"positive":"#7ef49e","neutral":"#f7dd8f","negative":"#f47e7e"}
                )
                fig2.update_layout(title_font_color="#bacaff", paper_bgcolor="#232a3b", plot_bgcolor="#232a3b", font_color="#e6eafe")
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
