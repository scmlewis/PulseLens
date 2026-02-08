import os
import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

st.set_page_config(
    page_title="PulseLens — Customer Pulse Analyzer",
    page_icon="📡",
    layout="wide"
)

# Header and style
try:
    with open(os.path.join(os.path.dirname(__file__), "static", "styles.css"), "r", encoding="utf-8") as fh:
        _css = fh.read()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
except Exception:
    # Fallback to inline minimal styling if file missing
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown('<style>.main-container{max-width:100vw}</style>', unsafe_allow_html=True)
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-banner">
    <h1>📡 PulseLens — Customer Pulse Analyzer</h1>
    <div class="desc">AI-powered customer pulse and aspect insights.</div>
</div>
""", unsafe_allow_html=True)

# Model and data
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_classifier():
    if 'classifier' not in st.session_state:
        st.session_state['classifier'] = load_zero_shot()
    return st.session_state['classifier']


def classify_batch(reviews, candidate_labels, classifier_getter, chunk_size=32, progress=None):
    """Classify reviews in chunks, returning a list of outputs per review.
    `classifier_getter` should be a callable returning the HF pipeline.
    Progress, if provided, must implement `.update(float)` where float is between 0.0 and 1.0.
    """
    if not reviews:
        return []
    cls = classifier_getter()
    results = []
    total = len(reviews)
    for i in range(0, total, chunk_size):
        chunk = reviews[i : i + chunk_size]
        try:
            outs = cls(chunk, candidate_labels=candidate_labels, multi_label=False)
        except Exception:
            # Fallback: classify one-by-one
            outs = []
            for item in chunk:
                try:
                    outs.append(cls(item, candidate_labels=candidate_labels, multi_label=False))
                except Exception:
                    outs.append({"labels": [], "scores": []})
        # Normalize to list of per-item outputs
        if isinstance(outs, dict):
            outs = [outs]
        results.extend(outs)
        if progress is not None:
            try:
                progress.update(min(1.0, (i + len(chunk)) / total))
            except Exception:
                pass
    return results
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
    # Map sentiment labels + confidence to a 1-5 star rating
    if sentiment == "positive":
        if score >= 0.9:
            return 5
        elif score >= 0.75:
            return 4
        elif score >= 0.6:
            return 3
        else:
            return 3
    elif sentiment == "neutral":
        return 3
    else:  # negative
        if score >= 0.85:
            return 1
        elif score >= 0.6:
            return 2
        else:
            return 2
def on_industry_select():
    sel = st.session_state.get('industry_select')
    if sel and sel != "-- Select industry --" and sel in GROUPED_ASPECTS:
        st.session_state['aspects_select'] = GROUPED_ASPECTS[sel]
        st.session_state['review_text'] = random.choice(SAMPLE_COMMENTS)

def set_sample():
    st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)
def clear_text():
    st.session_state["review_text"] = ""

# UI helpers for aspects
def _all_aspects():
    out = []
    for v in GROUPED_ASPECTS.values():
        out.extend(v)
    # preserve order, unique
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def append_aspect(key, aspect):
    cur = st.session_state.get(key, []) or []
    if aspect not in cur:
        st.session_state[key] = cur + [aspect]

def append_to_both(aspect):
    append_aspect('aspects_select', aspect)
    append_aspect('batch_aspects_select', aspect)

tab1, tab2, tab3 = st.tabs(["💬 Single Review", "📊 Batch Reviews", "❓ About & Help"])

# --- Single Review Tab ---
with tab1:
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    # ensure aspect multiselect state exists
    if 'aspects_select' not in st.session_state:
        st.session_state['aspects_select'] = []
    if 'industry_select' not in st.session_state:
        st.session_state['industry_select'] = ''
    st.markdown('<span style="color:#8eaffc;font-size:1.07em;font-weight:700;display:block;margin-bottom:0.09em;">💬 Enter a review</span>', unsafe_allow_html=True)
    # Wrap single-review inputs in a form to allow batching of inputs before inference
    with st.form(key='single_review_form'):
        text = st.text_area("", height=120, key="review_text", label_visibility="collapsed")
        st.markdown('<span style="color:#85e9ff;font-size:1.02em;font-weight:700;display:block;margin-bottom:0.02em;">🔎 Industry & Aspects</span>', unsafe_allow_html=True)
        col_ind, col_as = st.columns([1,2])
        with col_ind:
            industries = ["-- Select industry --"] + list(GROUPED_ASPECTS.keys())
            st.selectbox("Industry (preset)", industries, key='industry_select', label_visibility='collapsed', on_change=on_industry_select)
        with col_as:
            st.multiselect("Choose aspects (searchable)", options=_all_aspects(), default=st.session_state.get('aspects_select', []), key='aspects_select', label_visibility='collapsed')
        submit = st.form_submit_button(label="🚦 Classify Now")
    
    # Buttons outside the form (Streamlit 1.37+ requires callbacks to be outside forms)
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 0.13em; margin-bottom: 0.13em;">', unsafe_allow_html=True)
    st.button("✨ Generate Sample", on_click=set_sample, key="gen_sample_btn")
    st.button("🧹 Clear", on_click=clear_text, key="clear_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        # Run classification after form submit
        if not st.session_state.get('review_text', '').strip():
            st.error("Please enter a review.")
        elif not st.session_state.get('aspects_select'):
            st.error("Please select at least one aspect.")
        else:
            with st.spinner("🔄 Classifying… Please wait."):
                aspect_list = st.session_state.get('aspects_select', [])
                try:
                    cls = get_classifier()
                    sentiment_result = cls([st.session_state.get('review_text')], candidate_labels=SENTIMENT_LABELS)
                    if isinstance(sentiment_result, list):
                        sentiment_result = sentiment_result[0]
                    aspect_result = cls([st.session_state.get('review_text')], candidate_labels=aspect_list, multi_label=True)
                    if isinstance(aspect_result, list):
                        aspect_result = aspect_result[0]
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    sentiment_result = None
                    aspect_result = None
                sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                if sentiment_result is not None:
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
    
    <b>How to Use:</b>
    <ul>
      <li><b>Single Review:</b> Type or paste a customer review into the input box, enter one or more aspects (separated by commas), and click <b>Classify Now</b>.</li>
      <li><b>Batch Reviews:</b> Upload a CSV file containing a <code>review</code> column, or paste multiple reviews (one per line) in the provided textbox. Enter your aspects, then click <b>Classify Batch</b>.</li>
      <li>See instant results for sentiment (positive, neutral, negative), a 1-5 star conversion, and aspect alignment. Batch mode includes charts and downloadable CSV.</li>
    </ul>
    <b>What do "Sentiment" and "Rating" mean?</b>
    <ul>
      <li><b>Sentiment:</b> The AI classifies the overall tone of the review as positive 😊, neutral 😐, or negative 😞.</li>
      <li><b>Star Rating:</b> Sentiment confidence is mapped to 1-5 stars, offering a familiar, quick-glance metric.</li>
      <li><b>Aspects:</b> For each chosen aspect (like "service" or "price"), the AI gives a relevance score showing how much that topic drives the review's sentiment.</li>
    </ul>
    <b>How are results generated?</b>
    <ul>
      <li>AI-powered zero-shot classification ("facebook/bart-large-mnli") is used for all sentiment and aspect detection.</li>
      <li>No templates: the model uses deep language understanding for high accuracy across topics.</li>
      <li>Supports both short and long reviews, and any custom aspects you define.</li>
    </ul>
    <hr style="border:1px solid #282a39; margin:2em 0 1em 0;">
    <div style="background:#252d40; border-radius:12px; padding:1.7em 1.2em 1em 1.2em; margin-bottom:2em; box-shadow: 0 1.5px 12px #1b263433">
      <span class="suggested-aspects-title">Suggested Aspects (by Industry):</span>
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
    </div>
    <b>Questions?</b>
    <ul>
      <li>CSV uploads require a column titled exactly <b>review</b>.</li>
      <li>Choose aspects ("service", "value", etc.) that fit the context of your feedback.</li>
      <li>All processing is instant and secure—no text is sent to external servers or stored.</li>
      <li>For further help or feedback, please contact the developer.</li>
    </ul>
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
    if 'batch_aspects_select' not in st.session_state:
        st.session_state['batch_aspects_select'] = []
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
    st.multiselect("Choose aspects for batch (searchable)", options=_all_aspects(), default=st.session_state.get('batch_aspects_select', []), key='batch_aspects_select', label_visibility='collapsed')
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
        if not st.session_state.get('batch_aspects_select'):
            st.error("Please enter at least one aspect.")
        elif not reviews:
            st.error("Please upload CSV or enter reviews.")
        else:
            with st.spinner("🔄 Classifying batch reviews… Please wait."):
                results = []
                aspect_list = st.session_state.get('batch_aspects_select', [])
                try:
                    # chunked classification with visual progress
                    p1 = st.progress(0.0)
                    class _Prog:
                        def __init__(self, bar):
                            self.bar = bar
                        def update(self, v):
                            try:
                                self.bar.progress(v)
                            except Exception:
                                pass
                    sentiment_batch = classify_batch(reviews, SENTIMENT_LABELS, get_classifier, chunk_size=32, progress=_Prog(p1))
                    p2 = st.progress(0.0)
                    aspects_batch = classify_batch(reviews, aspect_list, get_classifier, chunk_size=16, progress=_Prog(p2))
                except Exception as e:
                    st.error(f"Model inference failed during batch processing: {e}")
                    sentiment_batch = []
                    aspects_batch = []

                for i, r in enumerate(reviews):
                    if i < len(sentiment_batch):
                        sres = sentiment_batch[i]
                    else:
                        sres = {"labels": ["neutral"], "scores": [0.0]}
                    if i < len(aspects_batch):
                        ares = aspects_batch[i]
                    else:
                        ares = {"labels": [""], "scores": [0.0]}
                    stars = sentiment_to_stars(sres['labels'][0], sres['scores'][0])
                    results.append({
                        "review": r,
                        "sentiment": sres["labels"][0],
                        "sentiment_score": sres["scores"][0],
                        "star_rating": stars,
                        "top_aspect": ares["labels"][0],
                        "aspect_score": ares["scores"][0]
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

# Re-enter About tab to render interactive suggested-aspect buttons
with tab3:
    st.markdown('**Suggested Aspects (click to add to inputs)**')
    for group, chips in GROUPED_ASPECTS.items():
        st.markdown(f"**{group}**")
        # render chips in rows of up to 6
        for i in range(0, len(chips), 6):
            row = chips[i:i+6]
            cols = st.columns(len(row))
            for j, chip in enumerate(row):
                safe_key = f"chip_{group}_{chip}".replace(' ', '_').replace('/', '_')
                cols[j].button(chip, key=safe_key, on_click=append_to_both, args=(chip,))
