import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(
    page_title="Customer Feedback Sentiment & Aspect Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Modern Wide-Mode and Responsive CSS ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
<style>
.stApp { background: #181c27 !important; font-family: 'Inter', sans-serif !important; color: #f3f6fb !important;}
.main-container { max-width: 1180px; margin: auto;}
.header-banner {
    background: linear-gradient(90deg,#4F8BF9 65%, #2D5AAB 100%);
    border-radius:1.15rem; padding:1.17em 2.2em 1.13em 2.2em; box-shadow:0 2px 16px #23274629;
    text-align:center; border-bottom: 4px solid #43a2ff; margin-bottom: 2em;
}
.header-banner h1 { font-size: 2.07em; font-weight: 900; margin: 0 0 0.14em 0; color: #fff; line-height:1.1; letter-spacing:-1px;}
.header-banner .desc { font-weight: 500; font-size: 1.17em; color: #e2ebff; margin-top:0.18em; }
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
    padding: 0.7em 2.3em; font-size: 1.14em !important;border:none;box-shadow:0 1px 10px #18113319;
    margin: 0 0.4em 0.55em 0.4em;
    display:inline-block;
    transition: all 0.22s;
}
.stButton > button:hover, .stButton>button:focus {
    background: linear-gradient(90deg,#3970e8,#61a3fd) !important; color: #fff !important;}
.stTextInput > div>input, .stTextArea textarea, .stSelectbox>div>div {
    background: #22283a !important;
    border-radius: 8px !important;
    color: #e3f2ff !important;
    border: 2px solid #3a4ece !important;
    font-size: 1.13em !important;
    font-weight: 500 !important;
}
.stTextInput>label, .stTextArea>label, .stSelectbox>label {
    color: #8ccaed !important; font-size: 1.14em; font-weight: 900!important;
}
.stTextInput > div>input:focus, .stTextArea textarea:focus, .stSelectbox>div>div:focus {
    border-color: #61a3fd !important; box-shadow: 0 0 8px #395fa7 !important;}
.card { background: #232a3b; border-radius: 13px; box-shadow: 0 5px 20px #1c223510;
    padding: 1.5rem 2.1rem 1.43rem 2.1rem; margin-bottom: 1.5rem;}
.stDataFrame >div>div { border-radius: 11px !important; box-shadow:0 0 14px #151d2e;}
.st-expander { background: #202b44 !important; border-radius: 14px !important; color: #e3eefd;}
[data-testid="stSidebar"] { background: #181c27 !important; min-width:300px; width:350px;}
/* Sidebar - responsive grid, flat chips, fits flex */
.sidebar-how-header {
    font-size: 1.24em; font-weight: 900; color: #7fc5ff; margin-bottom: 0.25em; margin-top: 0.12em; letter-spacing: 0.01em;
}
.sidebar-aspect-grid {
    display: flex; flex-wrap: wrap; gap:0.9em 1.35em; margin-bottom: 1.2em; margin-top:0.13em;
    justify-content:left;
}
.sidebar-aspect-group {
    background: #232a3b;
    border-radius:7px;
    margin-bottom:0.11em; min-width:unset; max-width:900px; display:flex;align-items:center; flex-wrap: wrap;
    box-shadow: 0 0.5px 2px #23306027;
    padding: 0.36em 1.09em 0.37em 0.81em;
}
.aspect-group-title {
    font-size:1.06em; color:#8eb5ff; font-weight:700; margin-right: 0.95em; margin-bottom:0.12em;
    flex-shrink:0;
    letter-spacing:0.01em;
    width:110px;
}
.aspect-chip {
    display:inline-block;background:#293053;color:#e7eefe;border-radius: 6.5px;
    padding: 0.23em 0.79em;margin:0.13em 0.35em 0.13em 0;font-size:0.97em;letter-spacing:0.01em;
}
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
.aspect-score {font-size:1.03em;font-weight:500;margin-left:0.27em; color:#77ebfa;}
/* Adjust main area when sidebar is collapsed */
@media (max-width: 1120px) {
    .main-container { max-width: 97vw !important; }
}
[data-testid="stSidebarContent"]:empty { min-width:0!important; padding:0 !important; margin: 0 !important; display:none!important; }
[data-testid="stSidebar"] { padding:0; min-width:unset; transition:0.22s all;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---- HEADER: CENTERED, RESPONSIVE ----
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

# -- SIDEBAR: "How to use" header, Help box, flat aspects grid in expander --
st.sidebar.markdown('<div class="sidebar-how-header">How to use</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style='background: #202c45; border-left: 4px solid #4F8BF9; border-radius: 10px; padding: 1em 1.3em 1em 1em;
    margin-bottom: 1.12em; color: #bae2ff; font-size: 1.09em; line-height: 1.7;'>
        <span style="font-size:1.13em; margin-right:0.33em;">💡</span>
        Enter a review, choose aspects (comma-separated), then classify. For batch analysis, upload a CSV or paste reviews below. Results will show live sentiment, aspect rating, and charts. All results are processed securely and instantly in your browser.
    </div>
    """,
    unsafe_allow_html=True
)
with st.sidebar.expander("Suggested Aspects", expanded=False):
    # Flat, responsive grid, each group horizontally aligned
    grid_html = '<div class="sidebar-aspect-grid">'
    for group, aspects in GROUPED_ASPECTS.items():
        grid_html += '<div class="sidebar-aspect-group">'
        grid_html += f'<span class="aspect-group-title">{group}</span>'
        for asp in aspects:
            grid_html += f'<span class="aspect-chip">{asp}</span>'
        grid_html += '</div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

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

# ---- MAIN: TABBED NAVIGATION including About/Help ----

TABS = ["💬 Single Review", "📊 Batch Reviews", "❓ About & Help"]
tab1, tab2, tab3 = st.tabs(TABS)

with tab1:
    # Centered, large adjacent action buttons
    st.markdown('<div style="display:flex;justify-content:center;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1: st.button("✨ Generate Sample", on_click=lambda: st.session_state.update({"review_text": random.choice(SAMPLE_COMMENTS)}))
    with col2: st.button("🧹 Clear", on_click=lambda: st.session_state.update({"review_text": ""}))
    st.markdown('</div>', unsafe_allow_html=True)
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    st.markdown('<span style="color:#8eaffc;font-size:1.1em;font-weight:700;"><b>💬 Enter a review</b></span>', unsafe_allow_html=True)
    text = st.text_area("", height=120, key="review_text")
    st.markdown('<span style="color:#85e9ff;font-size:1.08em;font-weight:700;"><b>🔎 Aspects/Categories (comma-separated)</b></span>', unsafe_allow_html=True)
    aspects = st.text_input("", value="", key="aspects_text")
    st.markdown('<div style="display:flex;justify-content:center;margin-top:0.7em;margin-bottom:0.7em;">', unsafe_allow_html=True)
    if st.button("🚦 Classify Now", key="classify_single_btn"):
        if not text.strip():
            st.error("Please enter a review.")
        elif not aspects.strip():
            st.error("Please enter at least one aspect.")
        else:
            with st.spinner("🔄 Classifying… Please wait."):
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
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown(
        "<div style='background: #22304a; border-radius: 11px; padding:0.78em 1.22em 0.86em 1.22em; color:#e3f1fe; font-size:1.10em;margin-bottom:1.09em;'>"
        "<b>Instructions:</b> Upload a UTF-8 CSV file with a column <span style='background:#222c41;color:#53ffb2;border-radius:5px;padding:1.5px 5px;font-size:1em;'>review</span> or paste reviews (one per line) below.<br>"
        "Enter aspects and press 'Classify Batch' to view results and charts.</div>",
        unsafe_allow_html=True
    )
    with st.expander("Batch Input Options"):
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_file = st.file_uploader("🗂️ <b>CSV Upload</b>", type=["csv"], key="batch_csv")
        with col2:
            st.markdown('<span style="color:#8eaffc;font-size:1.05em;font-weight:700;"><b>📋 Paste reviews (one per line)</b></span>', unsafe_allow_html=True)
            manual_text = st.text_area("", height=120, key="batch_manual_text")
    st.markdown('<span style="color:#85e9ff;font-size:1.06em;font-weight:700;"><b>🔎 Aspects/Categories for batch</b></span>', unsafe_allow_html=True)
    aspects = st.text_input("", value="", key="batch_aspects_text")
    reviews = []
    if (csv_file is not None):
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
        st.markdown(
            f"<span style='font-size:1.08em;color:#8eaffc;font-weight:700;'>Loaded {len(reviews)} reviews</span>", unsafe_allow_html=True
        )
        st.write("Sample Reviews", pd.DataFrame({"review": reviews[:5]}))
    st.markdown('<div style="display:flex;justify-content:center;margin-top:0.77em;margin-bottom:0.7em;">', unsafe_allow_html=True)
    if st.button("🚦 Classify Batch", key="classify_batch_btn"):
        if not aspects.strip():
            st.error("Please enter at least one aspect.")
        elif not reviews:
            st.error("Please upload CSV or enter reviews.")
        else:
            with st.spinner("🔄 Classifying batch… Please wait."):
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
                f'''<div class="output-card"><span style="font-size:1.14em;color:#9bc8ff;font-weight:900;">Batch Classification Results:</span></div>''',
                unsafe_allow_html=True
            )
            st.dataframe(results_df)
            st.markdown('<span style="font-size:1.07em;font-weight:800;color:#82b7ff;margin-top:0.5em;">Rating Distribution:</span>', unsafe_allow_html=True)
            pie_colors = ["#CEE6FF", "#ABCBFA", "#6B91F7", "#4E73CA", "#353A58"]
            fig1 = px.pie(results_df, names="star_rating", title="", color="star_rating",
                color_discrete_sequence=pie_colors)
            fig1.update_traces(textfont_color='black', marker=dict(line=dict(color='#232f3c', width=2)))
            fig1.update_layout(
                paper_bgcolor="#232a3b",
                plot_bgcolor="#232a3b",
                font_color="#e6eafe",
                margin=dict(l=8, r=8, t=8, b=8),
                legend=dict(orientation="v", x=1, y=0.7),
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('<span style="font-size:1.07em;font-weight:800;color:#82b7ff;margin-top:0.6em;">Sentiment Analysis:</span>', unsafe_allow_html=True)
            bar_colors = ['#50e396','#F9CE1D','#f97a77']
            fig2 = px.bar(results_df, x="sentiment", title="", color="sentiment",
                color_discrete_map={"positive":bar_colors[0],"neutral":bar_colors[1],"negative":bar_colors[2]})
            fig2.update_layout(
                paper_bgcolor="#232a3b",
                plot_bgcolor="#232a3b",
                font_color="#e6eafe",
                margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig2, use_container_width=True)
            csv_result = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_result,
                file_name="classification_results.csv",
                mime="text/csv"
            )
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown(
        """
        <div style="max-width: 770px; margin: 1.7em auto 1.1em auto; background:#232a3b; border-radius:11px; padding:1.9em 2em 1.3em 2em;">
        <h2 style="text-align:center; color:#8eaffc; margin-bottom:0.55em;">❓ About & Help</h2>
        <b>Purpose:</b> Easily analyze customer feedback or reviews to extract sentiment, key aspects, and visualize results in one streamlined platform.<br><br>
        <b>How to Use:</b>
        <ul>
        <li><b>Single Review</b>: Type or paste a customer review, enter one or more aspects (separated by commas), and click <b>Classify Now</b>.</li>
        <li><b>Batch Reviews</b>: Upload a CSV file with a <span style="background:#222c41;color:#53ffb2;padding:1.5px 7px;border-radius:5px;">review</span> column, or paste multiple reviews (one per line). Enter aspects and click <b>Classify Batch</b>.</li>
        <li>Results show the main sentiment detected, a 1-5 star mapping, and aspect alignment. Batch mode adds charts and downloadable results.</li>
        <li>Use the <b>Suggested Aspects</b> expander for common aspect ideas per industry, or create your own.</li>
        </ul>
        <b>Know-How:</b>
        <ul>
        <li>Sentiment is found using a robust general-purpose transformer (facebook/bart-large-mnli).</li>
        <li>Aspect scores tell how relevant each topic is in the text, so you can see what's driving good or bad reviews.</li>
        <li>Batch mode is ideal for customer surveys, e-commerce feedback, product evaluations, etc.</li>
        <li>Your data is processed on-device only—it's instant and private.</li>
        </ul>
        <hr style="border:1px solid #282a39; margin:1.4em 0;">
        <b>Questions?</b>
        <ul>
        <li>Check your input: Is the CSV header exactly <b>review</b>? Are aspects separated by commas?</li>
        <li>For more help or feedback, please contact the developer.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<hr><div style='color:#8aa2ff;font-size:1em;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</div>",
    unsafe_allow_html=True
)
