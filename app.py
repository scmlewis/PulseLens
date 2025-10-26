import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(
    page_title="Customer Feedback Sentiment & Aspect Classifier",
    page_icon="🧠",
    layout="wide"
)

# Header and style
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
/* Aspects grid and card styles - now in tab, not sidebar */
.suggested-aspects-title { font-size: 1.28em; color: #86c6f6; font-weight: 900; margin-bottom:0.4em; }
.suggested-aspect-grid { display: flex; flex-wrap: wrap; gap:1.7em 1em; padding:0 0.3em; }
.suggested-aspect-group { background: #232a3b; border-radius:10px; margin-bottom:0.09em;
    margin-top:0.13em; padding: 0.9em 1.19em 0.75em 1.1em; box-shadow: 0 0.5px 6px #1d223927;}
.suggested-group-title { font-size: 1.12em; color:#79b3f9; font-weight:900; margin-bottom:0.36em; }
.suggested-chips-row { display: flex; flex-wrap: wrap; gap:0.28em 0.49em; margin-top:0.01em;}
.suggested-chip { display:inline-block;background:#293053;color:#e7eefe;border-radius:7px;
    padding: 0.22em 1em;margin:0 0.06em 0.14em 0;font-size:1em;font-weight:600; letter-spacing:0.01em;}
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

# Model and data
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

# --- Single Review Tab ---
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

# --- Batch Reviews Tab (unchanged logic, as prior robust version) ---
# ... [Paste in batch reviews code from before] ...

# --- About/help tab - with suggested aspects cards ---
with tab3:
    st.markdown("""
    <div style="max-width: 800px; margin: 1.2em auto 0 auto; background:#232a3b; border-radius:13px; padding:2em 2.4em 1.8em 2.4em;">
    <h2 style="text-align:center; color:#8eaffc; margin-bottom:0.5em;">❓ About & Help</h2>
    <div><b>Suggested Aspects (by Industry):</b></div>
    <div class="suggested-aspect-grid">
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">🍽️ Restaurant</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">food</span><span class="suggested-chip">service</span><span class="suggested-chip">ambience</span><span class="suggested-chip">price</span><span class="suggested-chip">delivery</span><span class="suggested-chip">staff</span><span class="suggested-chip">product quality</span>
            </div>
        </div>
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">💻 Electronics</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">battery</span><span class="suggested-chip">display</span><span class="suggested-chip">camera</span><span class="suggested-chip">performance</span><span class="suggested-chip">durability</span><span class="suggested-chip">shipping</span><span class="suggested-chip">support</span>
            </div>
        </div>
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">👗 Fashion</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">fit</span><span class="suggested-chip">material</span><span class="suggested-chip">style</span><span class="suggested-chip">comfort</span><span class="suggested-chip">design</span><span class="suggested-chip">price</span>
            </div>
        </div>
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">🛒 Supermarket</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">freshness</span><span class="suggested-chip">variety</span><span class="suggested-chip">checkout</span><span class="suggested-chip">customer service</span><span class="suggested-chip">packaging</span><span class="suggested-chip">speed</span>
            </div>
        </div>
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">📚 Books</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">plot</span><span class="suggested-chip">characters</span><span class="suggested-chip">writing</span><span class="suggested-chip">pacing</span><span class="suggested-chip">ending</span><span class="suggested-chip">value</span>
            </div>
        </div>
        <div class="suggested-aspect-group">
            <div class="suggested-group-title">🏨 Hotel</div>
            <div class="suggested-chips-row">
                <span class="suggested-chip">cleanliness</span><span class="suggested-chip">location</span><span class="suggested-chip">amenities</span><span class="suggested-chip">room</span><span class="suggested-chip">wifi</span><span class="suggested-chip">maintenance</span>
            </div>
        </div>
    </div>
    <!-- (You can put the rest of your About/help prose below) -->
    <hr style="border:1px solid #282a39; margin:2em 0 1em 0;">
    <b>How to Use:</b>
    ... [rest of your help/instructions as you like] ...
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<hr><div style='color:#8aa2ff;font-size:1em;'>Model: facebook/bart-large-mnli (Meta, Hugging Face)</div>",
    unsafe_allow_html=True
)

with tab2:
    if 'uploaded_filename' not in st.session_state:
        st.session_state['uploaded_filename'] = ''
    st.markdown(
        "<div style='background: #22304a; border-radius: 11px; padding:0.73em 1.22em 0.85em 1.22em; color:#e3f1fe; font-size:1.07em;margin-bottom:1.03em;'>"
        "<b>Instructions:</b> Upload a UTF-8 CSV file with a column <span style='background:#222c41;color:#53ffb2;border-radius:5px;padding:1.5px 5px;font-size:1em;'>review</span> or paste reviews (one per line) below.<br>"
        "Enter aspects and press 'Classify Batch' to view results and charts.</div>",
        unsafe_allow_html=True
    )
    with st.expander("Batch Input Options"):
        col1, col2 = st.columns([1, 1])
        uploaded = None
        with col1:
            st.markdown('<span style="color:#f9d66e;font-size:1.08em;font-weight:700;display:block;margin-bottom:0.04em;">🗂️ CSV Upload</span>', unsafe_allow_html=True)
            uploaded = st.file_uploader("", type=["csv"], key="batch_csv", help="Upload your .csv file with review column.")
        with col2:
            st.markdown('<span style="color:#8eaffc;font-size:1.05em;font-weight:700;display:block;margin-bottom:0.04em;">📋 Paste reviews (one per line)</span>', unsafe_allow_html=True)
            manual_text = st.text_area("", height=120, key="batch_manual_text", label_visibility="collapsed")

    st.markdown('<span style="color:#85e9ff;font-size:1.02em;font-weight:700;display:block;margin-bottom:0.01em;">🔎 Aspects/Categories for batch</span>', unsafe_allow_html=True)
    aspects = st.text_input("", value="", key="batch_aspects_text", label_visibility="collapsed")
    reviews = []
    uploaded_filename = ''
    if uploaded is not None:
        if uploaded != st.session_state.get("uploaded_filename"):
            st.session_state["uploaded_filename"] = uploaded
            uploaded_filename = uploaded.name
            try:
                dataframe = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                dataframe = pd.read_csv(uploaded, encoding="latin1")
            if 'review' in dataframe.columns:
                rec_count = len(dataframe['review'].dropna())
                st.info(f"✅ File uploaded: '{uploaded_filename}', {rec_count} reviews detected.")
                reviews = dataframe['review'].dropna().astype(str).tolist()
            else:
                st.warning("⚠️ No column named 'review' found in uploaded CSV.")
        else:
            try:
                dataframe = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                dataframe = pd.read_csv(uploaded, encoding="latin1")
            if 'review' in dataframe.columns:
                reviews = dataframe['review'].dropna().astype(str).tolist()
    elif manual_text and manual_text.strip():
        reviews = [line.strip() for line in manual_text.split("\n") if line.strip()]

    if reviews:
        st.markdown(f"<span style='font-size:1.07em;color:#8eaffc;font-weight:700;'>Loaded {len(reviews)} reviews</span>", unsafe_allow_html=True)
        st.write("Sample Reviews", pd.DataFrame({"review": reviews[:5]}))

    st.markdown('<div style="display:flex;justify-content:center;margin-top:0.6em;margin-bottom:0.6em;">', unsafe_allow_html=True)
    if st.button("🚦 Classify Batch", key="classify_batch_btn2"):
        if not aspects.strip():
            st.error("Please enter at least one aspect.")
        elif not reviews:
            st.error("Please upload CSV or enter reviews.")
        else:
            with st.spinner("🔄 Classifying batch reviews… Please wait."):
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
            st.success("✅ Batch classification completed!")
            results_df = pd.DataFrame(results)
            ordered = pd.CategoricalDtype([1,2,3,4,5], ordered=True)
            results_df['star_rating'] = results_df['star_rating'].astype(ordered)
            st.markdown(f'''<div class="output-card"><span style="font-size:1.14em;color:#9bc8ff;font-weight:900;">Batch Classification Results:</span></div>''', unsafe_allow_html=True)
            st.dataframe(results_df)
            st.markdown('<span style="font-size:1.07em;font-weight:800;color:#82b7ff;margin-top:0.5em;">Rating Distribution:</span>', unsafe_allow_html=True)
            pie_colors = ["#E05555", "#FAAA28", "#4BA3FE", "#789A37", "#60CAAE"]
            fig1 = px.pie(results_df, names="star_rating", title="", color="star_rating",
                category_orders={"star_rating":[1,2,3,4,5]},
                color_discrete_sequence=pie_colors)
            fig1.update_traces(textfont_color='white', marker=dict(line=dict(color='#232f3c', width=2)))
            fig1.update_layout(
                paper_bgcolor="#232a3b",
                plot_bgcolor="#232a3b",
                font_color="#e6eafe",
                margin=dict(l=8, r=8, t=8, b=8),
                legend=dict(orientation="v", x=1, y=0.7),
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('<span style="font-size:1.07em;font-weight:800;color:#82b7ff;margin-top:0.5em;">Sentiment Analysis:</span>', unsafe_allow_html=True)
            bar_colors = ['#50e396','#F9CE1D','#f97a77']
            fig2 = px.bar(results_df, x="sentiment", title="", color="sentiment",
                color_discrete_map={"positive":bar_colors[0],"neutral":bar_colors[1],"negative":bar_colors[2]}
            )
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
