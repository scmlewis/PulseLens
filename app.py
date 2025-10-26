import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

# ---- CSS block: Modern dashboard/Inter/gradient/header/nav/buttons/helpbox/cards/feedback/resp ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp { background: #181c27 !important; font-family: 'Inter', sans-serif !important; }
.top-banner {
    width: 100vw;
    margin-left: -3.5vw;
    background: linear-gradient(90deg, #4F8BF9 65%, #2D5AAB 100%);
    padding: 2.1rem 0 1.6rem 2.3rem;
    color: #fff;
    box-shadow: 0 3px 16px #191c2859;
    border-radius: 0 0 1.6rem 1.6rem;
    margin-bottom: 2.15em;
    border-bottom: 4px solid #43a2ff;
}
.top-banner h1 { font-size: 2.2em; font-weight: 800; margin: 0; color: #fff; }
.top-banner .desc { font-weight: 430; font-size: 1.12em; color: #dae7ff; }
.nav-row {
    width: 100%;
    display: flex; flex-wrap: wrap; justify-content: flex-start; align-items: flex-end;
    margin-bottom: 0.44em; gap: min(1vw, 16px);
    margin-top: 0.4em;
}
.nav-btn {
    display: flex; align-items: center; justify-content: center;
    border: none; cursor: pointer;
    border-radius: 999px;
    font-size: 1.15em; font-weight: 700;
    padding: 0.8em 1.7em;
    margin-right: 0.3em; margin-top:0.12em; margin-bottom:0.14em;
    background: linear-gradient(90deg, #292d43, #212644 100%);
    color: #bfdcff;
    box-shadow: 0 1.5px 14px #1b1c237a;
    transition: all 0.17s;
}
.nav-btn[aria-current="page"] {
    background: linear-gradient(90deg,#5dc7ff,#4f80f9 70%, #5f95fc 100%);
    color: #fff;
    box-shadow: 0 3px 16px #43a2ff34;
    filter: brightness(1.08);
}
.nav-btn:hover, .nav-btn:active {
    background: linear-gradient(90deg,#366fd9,#575fcf);
    color: #fff;
}
@media(max-width: 650px) {
    .top-banner h1 { font-size: 1.18em; }
    .nav-btn { font-size: 0.98em; padding: 0.7em 0.9em;}
}
.help-box {
    background: #23273d;
    border-left: 5px solid #56aaff;
    border-radius: 8px;
    padding: 1.1em 1.3em 1.05em 1.2em;
    color: #bfcfff;
    margin-bottom: 1.6em;
    font-size: 1.08em;
    line-height: 1.7;
    box-shadow: 0 2px 8px #14182523;
}
.card {
    background: #232a3b;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(50,60,75,0.13);
    padding: 2rem 2.1rem 1.5rem 2rem;
    margin-bottom: 2.15rem;
}
.stTabs [data-baseweb='tab-list'] { background: none !important; }
.stTabs [data-baseweb='tab'] { font-size: 1.13em; font-weight: 700; color: #c9e0ff !important; background: none;}
.stTabs [data-baseweb='tab'][aria-selected='true'] { color: #fff !important; border-bottom:3px solid #4F8BF9; }
.stButton > button {
    background: linear-gradient(90deg,#32449b,#485cdd);
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.73em 2.11em !important;
    font-size: 1.06em !important;
    margin: 0.19em 1em 0.3em 0;
    border:none;
    box-shadow: 0 1px 8px #202a3d51;
}
.stButton > button:hover, .stButton > button:focus {
    background: linear-gradient(90deg,#3970e8,#61a3fd) !important;
    color: #fff !important;
}
.stTextInput > div>input, .stTextArea textarea, .stSelectbox>div>div {
    background: #22283a !important;
    border-radius: 8px !important;
    color: #e3f2ff !important;
    border: 2px solid #3a4ece !important;
    font-size: 1.13em !important;
    font-weight: 500 !important;
}
.stTextInput > div>input:focus, .stTextArea textarea:focus, .stSelectbox>div>div:focus {
    border-color: #61a3fd !important;
    box-shadow: 0 0 8px #395fa7 !important;
}
.stDataFrame > div > div { border-radius: 12px !important; box-shadow:0 0 14px #171d2e;}
.st-expander { background: #1b2134 !important; border-radius: 15px !important; color: #c3cff8;}
/* Sidebar sections & lists */
[data-testid="stSidebar"] { background: #181c27 !important;}
.sb-section { margin-bottom: 1.17em; }
.sb-hdr { color: #8ebafa !important; font-weight: bold; font-size: 1.02em; margin-bottom:0.17em;}
.sb-list { padding-left:1.1em; margin-top:0.17em; margin-bottom:0.54em;}
.sb-list li { color: #e8eaff; font-size: 1em; margin-bottom:0.11em; line-height: 1.45;}
hr.sb-hr { border: none; border-top: 1px solid #23305a; margin: 0.77em 0 0.85em 0;}
/* Feedback animated slide-in */
.stAlert {animation:slidein 0.5s;}
@keyframes slidein { from { opacity:0;transform: translateY(-14px);} to { opacity: 1;transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown(
    """
    <div class='top-banner'>
        <h1>🧠 Customer Feedback Sentiment & Aspect Classifier</h1>
        <div class='desc'>Modern, AI-powered feedback analytics for all industries.</div>
    </div>
    """, unsafe_allow_html=True
)
# Simulated horizontal nav bar
nav_buttons = [
    ("🏠 Welcome", "welcome"),
    ("💬 Single Review", "single"),
    ("📊 Batch Reviews", "batch"),
    ("📤 Upload Data", "upload"),
    ("📈 Visualize", "visualize"),
    ("🤖 AI Chat", "chat")
]
nav_query = st.query_params if hasattr(st, 'query_params') else {}
current_section = nav_query.get("section", "single")  # default for now, might want to adjust logic
nav_markup = '<div class="nav-row">'
for label, section in nav_buttons:
    selected = (section == current_section)
    nav_markup += f'<button class="nav-btn" {"aria-current=page" if selected else ""}>{label}</button>'
nav_markup += "</div>"
st.markdown(nav_markup, unsafe_allow_html=True)

######## Sidebar with improved list
GROUPED_ASPECTS = {
    "🍽️ Restaurant": ["food", "service", "ambience", "price", "delivery", "staff", "product quality"],
    "💻 Electronics": ["battery", "display", "camera", "performance", "durability", "shipping", "support"],
    "👗 Fashion": ["fit", "material", "style", "comfort", "design", "price"],
    "🛒 Supermarket": ["freshness", "variety", "checkout", "customer service", "packaging", "speed"],
    "📚 Books": ["plot", "characters", "writing", "pacing", "ending", "value"],
    "🏨 Hotel": ["cleanliness", "location", "amenities", "room", "wifi", "maintenance"]
}
def render_grouped_aspects():
    st.sidebar.markdown('<div class="sb-section"><span class="sb-hdr">Suggested Aspects</span></div>', unsafe_allow_html=True)
    for group, aspects in GROUPED_ASPECTS.items():
        st.sidebar.markdown(
            f"<div class='sb-section'><span class='sb-hdr'>{group}</span><ul class='sb-list'>" +
            "".join(f"<li>{asp}</li>" for asp in aspects) +
            "</ul></div><hr class='sb-hr'>", unsafe_allow_html=True
        )
render_grouped_aspects()

# ---- Main content nav: Help box ----
st.markdown("""
<div class="help-box">
    <span style="font-size:1.15em; margin-right:0.38em;">💡</span>
    Enter a review, choose aspects (comma-separated), then classify. For batch analysis, upload a CSV or paste reviews below.
    Results will show live sentiment, aspect rating, and charts. All results are processed securely and instantly in your browser.
</div>
""", unsafe_allow_html=True)

# ---- Main content: Card container and workflow tabbed UI ----
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_zero_shot()
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

st.markdown("""<div class="card">""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["💬 Single Review", "📊 Batch Reviews"])
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
        "<div style='color:#7e8abb; font-size:1.08em; margin-bottom: 0.6em;'><b>Instructions:</b> Upload a UTF-8 CSV file named <code>review</code> or paste reviews (one per line) below. Enter aspects below and press 'Classify Batch'.</div>",
        unsafe_allow_html=True
    )
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
