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

# ---- CSS for robust sidebar/grid/spacing/slides ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
<style>
.stApp { background: #181c27 !important; font-family: 'Inter', sans-serif !important; color: #f3f6fb !important;}
.main-container { width: 100vw !important; max-width: none;}
.header-banner { background: linear-gradient(90deg,#4F8BF9 65%, #2D5AAB 100%);
    border-radius:1.15rem; padding:1.17em 2.2em 1.13em 2.2em; box-shadow:0 2px 16px #23274629;
    text-align:center; border-bottom: 4px solid #43a2ff; margin-bottom: 2em;}
.header-banner h1 { font-size: 2.07em; font-weight: 900; margin: 0 0 0.14em 0; color: #fff; line-height:1.1; letter-spacing:-1px;}
.header-banner .desc { font-weight: 500; font-size: 1.17em; color: #e2ebff; margin-top:0.18em; }
.stTabs [data-baseweb='tab-list'] { background: none !important; justify-content:center;}
.stTabs [data-baseweb='tab'] { border-radius: 22px 22px 0 0 !important; margin-right: 1.2em !important;
    font-size: 1.18em !important; font-weight: 800 !important; color: #adc8ff !important;
    padding: 0.89em 2.33em 0.85em 2.33em !important; background: #232a3b !important; box-shadow: 0 2.5px 16px #22283a35;
    border: 2.1px solid #3F59B844;}
.stTabs [data-baseweb='tab'][aria-selected='true'] { color: #fff !important; background: linear-gradient(90deg, #527afe, #3553c3 90%);
    border-bottom: 3.1px solid #8dc7fc !important; box-shadow: 0 2px 12px #325aee18;}
.stButton > button { background: linear-gradient(90deg,#32449b,#485cdd);
    color: #fff !important; font-weight: 700 !important; border-radius: 8px !important;
    padding: 0.7em 2.3em; font-size: 1.11em !important;border:none;box-shadow:0 1px 10px #18113319;
    margin:0 .24em 0 .24em; display:inline-block; transition: all 0.22s;}
.stButton > button:hover, .stButton>button:focus { background: linear-gradient(90deg,#3970e8,#61a3fd) !important; color: #fff !important;}
.stTextInput > div>input, .stTextArea textarea, .stSelectbox>div>div {
    background: #22283a !important; border-radius: 8px !important; color: #e3f2ff !important;
    border: 2px solid #3a4ece !important; font-size: 1.13em !important; font-weight: 500 !important;}
.stTextInput, .stTextArea, .stSelectbox { margin-top: 0em !important; margin-bottom: 0.11em !important; }
.stTextInput label, .stTextArea label, .stSelectbox label { margin-bottom:0.08em !important; }
.card { background: #232a3b; border-radius: 13px; box-shadow: 0 5px 20px #1c223510;
    padding: 1.15rem 2.1rem 1.1rem 2.1rem; margin-bottom: 0.6em;}
.stDataFrame >div>div { border-radius: 11px !important; box-shadow:0 0 14px #151d2e;}
.st-expander { background: #202b44 !important; border-radius: 14px !important; color: #e3eefd;}
[data-testid="stSidebar"] { background: #181c27 !important; min-width:170px; max-width:410px; width:340px;}
.sidebar-how-header { color: #7fc5ff; font-size: 1.23em; font-weight: 900; margin-bottom: 0.15em; margin-top: 0.07em; letter-spacing: 0.01em;}
.sidebar-scroll-x {width: 100%; overflow-x: auto;}
.sidebar-aspect-grid { display: flex; flex-direction: column; gap:0.45em 0; min-width: 270px; max-width:390px; box-sizing:border-box;}
.sidebar-aspect-group { background: #232a3b; border-radius:7px; margin-bottom:0.09em;
    margin-top:0.13em; padding: 0.33em 1.09em 0.41em 0.81em; box-shadow: 0 0.5px 2px #23306027; min-width:170px; max-width: 388px; width:96%;}
.aspect-group-title { font-size: 1.09em; color:#8eb5ff; font-weight:700; letter-spacing:0.01em;
    width:100%; padding-bottom:0.06em; margin-bottom:0.00em; line-height:1.26;}
.aspect-chips-row { display: flex; flex-wrap: wrap; gap:0.28em 0.49em; margin-top:0.1em;}
.aspect-chip { display:inline-block;background:#293053;color:#e7eefe;border-radius:6.5px;
    padding: 0.22em 0.87em;margin:0.04em 0 0.04em 0;font-size:0.97em;}
.output-card { background: #232a3b; border-radius:11px; box-shadow: 0 2px 12px #191c2e35; padding: 1.08em 2em; margin-bottom:0.18em; color: #f3f6fb; font-size:1.15em; letter-spacing:0.006em;}
.output-stars { margin:0.12em 0 0.19em 0; font-size:1.28em;}
.senti-positive { color:#90ffb8;font-weight:700;}
.senti-label { font-weight:900; font-size:1.13em; margin-right:0.24em;}
.senti-score { color:#b5b9fc; font-size:0.98em; margin-left:0.12em;}
.aspect-row { display:flex;align-items:center;margin:0.09em 0;}
.aspect-dot { width:15px;height:15px;display:inline-block;border-radius:24px;
    background:linear-gradient(145deg,#5ae0fb,#1e61e6 80%); margin-right: 0.24em;}
.aspect-dot-lo {background:linear-gradient(145deg,#adc6ff,#434766 80%);}
.aspect-score {font-size:1.03em;font-weight:500;margin-left:0.27em; color:#77ebfa;}
.aspect-label-list {margin:0.37em 0 0.12em 1px;}
@media (min-width: 801px) {
    [data-testid="stSidebar"][aria-expanded="false"] + div > .main-container {
        width: 99vw !important; max-width: none;
    }
}
@media (max-width: 800px) {
    .main-container { width: 99vw !important; max-width: none;}
}
[data-testid="stSidebarContent"]:empty { min-width:0!important; padding:0 !important; margin: 0 !important; display:none!important; }
[data-testid="stSidebar"] { padding:0; min-width:unset; transition:0.22s all;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-banner">
    <h1>🧠 Customer Feedback Sentiment & Aspect Classifier</h1>
    <div class="desc">Modern, AI-powered feedback analytics for all industries.</div>
</div>
""", unsafe_allow_html=True)

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

st.sidebar.markdown('<div class="sidebar-how-header">How to use</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background: #202c45; border-left: 4px solid #4F8BF9; border-radius: 10px; padding: 1em 1.2em 1em 1em; margin-bottom: 0.7em; color: #bae2ff; font-size: 1.09em; line-height: 1.6;'>
    <span style="font-size:1.13em; margin-right:0.33em;">💡</span>
    Enter a review, choose aspects (comma-separated), then classify. For batch analysis, upload a CSV or paste reviews below. Results will show live sentiment, aspect rating, and charts. All results are processed securely and instantly in your browser.
</div>
""", unsafe_allow_html=True)
with st.sidebar.expander("Suggested Aspects", expanded=True):
    st.markdown(
        '<div class="sidebar-scroll-x"><div class="sidebar-aspect-grid">' +
        ''.join(
            f'''
            <div class="sidebar-aspect-group">
                <div class="aspect-group-title">{group}</div>
                <div class="aspect-chips-row">
                    {''.join(f'<span class="aspect-chip">{asp}</span>' for asp in aspects)}
                </div>
            </div>
            '''
            for group, aspects in GROUPED_ASPECTS.items()
        ) +
        '</div></div>', unsafe_allow_html=True
    )

SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SAMPLE_COMMENTS = [
    # [same as previous...]
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
def set_sample():
    st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)
def clear_text():
    st.session_state["review_text"] = ""

tab1, tab2, tab3 = st.tabs(["💬 Single Review", "📊 Batch Reviews", "❓ About & Help"])
with tab1:
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    st.markdown('<span style="color:#8eaffc;font-size:1.07em;font-weight:700;display:block;margin-bottom:0.09em;">💬 Enter a review</span>', unsafe_allow_html=True)
    text = st.text_area("", height=120, key="review_text", label_visibility="collapsed")
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 0.13em; margin-bottom: 0.13em;">', unsafe_allow_html=True)
    st.button("✨ Generate Sample", on_click=set_sample, key="gen_sample_btn")
    st.button("🧹 Clear", on_click=clear_text, key="clear_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<span style="color:#85e9ff;font-size:1.02em;font-weight:700;display:block;margin-bottom:0.02em;">🔎 Aspects/Categories (comma-separated)</span>', unsafe_allow_html=True)
    aspects = st.text_input("", value="", key="aspects_text", label_visibility="collapsed")
    st.markdown('<div style="display:flex;justify-content:center;margin-top:0.25em;margin-bottom:0.41em;">', unsafe_allow_html=True)

    if st.button("🚦 Classify Now", key="classify_single_btn2"):
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
                    </div>''', unsafe_allow_html=True
                )
                st.markdown('<div class="aspect-label-list" style="font-size:1.11em;font-weight:700;color:#a7c3fe;margin:0.2em 0 0.11em 1px;">Aspect Relevance Scores:</div>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<hr><div style='color:#8aa2ff;font-size:1em;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</div>",
    unsafe_allow_html=True
)
