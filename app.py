"""
PulseLens - Customer Feedback Analyzer
Lightweight Streamlit app - single deploy, no backend required.
"""
import os
import streamlit as st
import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

# ==================== DATA STRUCTURES ====================

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

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="PulseLens — Customer Pulse Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header and style
try:
    with open(os.path.join(os.path.dirname(__file__), "static", "styles.css"), "r", encoding="utf-8") as fh:
        _css = fh.read()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
except Exception:
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown('<style>.main-container{max-width:100vw}</style>', unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-banner">
    <h1>📡 PulseLens — Customer Pulse Analyzer</h1>
    <div class="desc">AI-powered sentiment & aspect classification using Hugging Face zero-shot learning.</div>
</div>
""", unsafe_allow_html=True)

# ==================== MODEL & CACHE ====================

@st.cache_resource
def load_zero_shot():
    """Load the zero-shot classification model once and cache it."""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_classifier():
    """Get or initialize the classifier in session state."""
    if 'classifier' not in st.session_state:
        st.session_state['classifier'] = load_zero_shot()
    return st.session_state['classifier']

# ==================== UTILITY FUNCTIONS ====================

def sentiment_to_stars(sentiment, score):
    """
    Map sentiment label + confidence score to 1-5 star rating.
    - Positive high-confidence (≥90%) → 5 stars
    - Positive medium-confidence (≥75%) → 4 stars
    - Neutral → 3 stars
    - Negative medium-confidence (≥60%) → 2 stars
    - Negative high-confidence (≥85%) → 1 star
    """
    if sentiment == "positive":
        if score >= 0.9:
            return 5
        elif score >= 0.75:
            return 4
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

def classify_batch(reviews, candidate_labels, classifier_getter, chunk_size=32, progress=None):
    """
    Classify reviews in chunks, returning a list of outputs per review.
    
    Args:
        reviews: List of review texts
        candidate_labels: List of aspects to classify
        classifier_getter: Callable that returns the HF pipeline
        chunk_size: Batch size for processing (default 32)
        progress: Optional progress bar object with .update(float) method
    
    Returns:
        List of classification outputs (one per review)
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
            outs = []
            for item in chunk:
                try:
                    outs.append(cls(item, candidate_labels=candidate_labels, multi_label=False))
                except Exception:
                    outs.append({"labels": [], "scores": []})
        
        if isinstance(outs, dict):
            outs = [outs]
        results.extend(outs)
        
        if progress is not None:
            try:
                progress.update(min(1.0, (i + len(chunk)) / total))
            except Exception:
                pass
    
    return results

# ==================== SESSION STATE CALLBACKS ====================

def on_industry_select():
    """Callback: when user selects an industry, populate aspects."""
    sel = st.session_state.get('industry_select')
    if sel and sel != "-- Select industry --" and sel in GROUPED_ASPECTS:
        st.session_state['aspects_select'] = GROUPED_ASPECTS[sel]

def set_sample():
    """Callback: load a random sample review."""
    st.session_state["review_text"] = random.choice(SAMPLE_COMMENTS)

def clear_text():
    """Callback: clear the review text."""
    st.session_state["review_text"] = ""

# ==================== MAIN APP ====================

# Helper function for all aspects
def get_all_aspects():
    out = []
    for v in GROUPED_ASPECTS.values():
        out.extend(v)
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Analyze a Review", "📊 Batch Reviews", "ℹ️ About & Help"])

# ===== TAB 1: SINGLE REVIEW ANALYSIS =====
with tab1:
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    if 'aspects_select' not in st.session_state:
        st.session_state['aspects_select'] = []
    if 'industry_select' not in st.session_state:
        st.session_state['industry_select'] = ''
    
    st.markdown("### 💬 Enter a Review")
    
    st.markdown("**🏭 Industry Preset (optional):**")
    industries = ["-- Select industry --"] + list(GROUPED_ASPECTS.keys())
    st.selectbox(
        "Choose an industry to auto-fill aspects:",
        industries,
        key='industry_select',
        label_visibility='collapsed',
        on_change=on_industry_select
    )
    
    with st.form(key='single_review_form'):
        st.markdown("**Review Text:**")
        text = st.text_area(
            "Enter review:",
            height=120,
            key="review_text",
            label_visibility="collapsed"
        )
        
        st.markdown("**Select Aspects:**")
        st.multiselect(
            "Choose aspects to analyze:",
            options=get_all_aspects(),
            default=st.session_state.get('aspects_select', []),
            key='aspects_select',
            label_visibility='collapsed'
        )
        
        submit = st.form_submit_button(label="🔍 Classify Now", type="primary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("✨ Load Sample", on_click=set_sample, key="gen_sample_btn")
    with col2:
        st.button("🧹 Clear", on_click=clear_text, key="clear_btn")

    if submit:
        if not st.session_state.get('review_text', '').strip():
            st.error("Please enter a review.")
        elif not st.session_state.get('aspects_select'):
            st.error("Please select at least one aspect.")
        else:
            with st.spinner("Analyzing sentiment and aspects..."):
                try:
                    cls = get_classifier()
                    
                    # Sentiment classification
                    sentiment_result = cls(
                        [st.session_state.get('review_text')],
                        candidate_labels=SENTIMENT_LABELS
                    )
                    if isinstance(sentiment_result, list):
                        sentiment_result = sentiment_result[0]
                    
                    # Aspect classification
                    aspect_list = st.session_state.get('aspects_select', [])
                    aspect_result = cls(
                        [st.session_state.get('review_text')],
                        candidate_labels=aspect_list,
                        multi_label=True
                    )
                    if isinstance(aspect_result, list):
                        aspect_result = aspect_result[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    sentiment = sentiment_result['labels'][0]
                    score = sentiment_result['scores'][0]
                    stars = sentiment_to_stars(sentiment, score)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment.title(), f"{score:.1%}")
                    with col2:
                        st.metric("Star Rating", "⭐" * stars, f"{stars}/5")
                    with col3:
                        st.metric("Confidence", f"{score:.1%}")
                    
                    st.markdown("---")
                    st.write("**Aspect Relevance Scores:**")
                    
                    aspect_df = pd.DataFrame({
                        'Aspect': aspect_result['labels'],
                        'Score': aspect_result['scores']
                    }).sort_values('Score', ascending=False)
                    
                    col_chart, col_table = st.columns([1, 1])
                    
                    with col_chart:
                        fig = px.bar(
                            aspect_df,
                            x='Score',
                            y='Aspect',
                            orientation='h',
                            color='Score',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_table:
                        display_df = aspect_df.copy()
                        display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"Classification failed: {e}")

# ===== TAB 2: BATCH REVIEWS =====
with tab2:
    st.markdown("### 📊 Batch Review Analysis")
    
    if 'batch_aspects_select' not in st.session_state:
        st.session_state['batch_aspects_select'] = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Upload CSV** (with `review` column):")
        uploaded_file = st.file_uploader("Choose CSV", type="csv", key="batch_file")
    
    with col2:
        st.markdown("**Or paste reviews** (one per line):")
        pasted_text = st.text_area("Paste reviews:", height=120, label_visibility='collapsed', key="batch_manual")
    
    st.markdown("**Select Aspects:**")
    st.multiselect(
        "Aspects to analyze:",
        options=get_all_aspects(),
        default=st.session_state.get('batch_aspects_select', []),
        key='batch_aspects_select',
        label_visibility='collapsed'
    )
    
    if st.button("⚙️ Classify Batch", type="primary", key="batch_btn"):
        reviews_list = []
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'review' not in df.columns:
                    st.error("CSV must have a 'review' column")
                else:
                    reviews_list = df['review'].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        elif pasted_text:
            reviews_list = [r.strip() for r in pasted_text.split('\n') if r.strip()]
        
        if not reviews_list:
            st.error("No reviews found. Please upload CSV or paste reviews.")
        elif not st.session_state.get('batch_aspects_select'):
            st.error("Please select at least one aspect.")
        else:
            with st.spinner("Classifying batch... This may take a minute."):
                try:
                    # Sentiment analysis
                    progress_bar = st.progress(0, text="Analyzing sentiment...")
                    sentiment_results = classify_batch(
                        reviews_list,
                        SENTIMENT_LABELS,
                        get_classifier,
                        chunk_size=32
                    )
                    progress_bar.empty()
                    
                    # Aspect analysis
                    progress_bar = st.progress(0, text="Analyzing aspects...")
                    aspect_results = classify_batch(
                        reviews_list,
                        st.session_state['batch_aspects_select'],
                        get_classifier,
                        chunk_size=32
                    )
                    progress_bar.empty()
                    
                    # Build results
                    batch_results = []
                    for i, review in enumerate(reviews_list):
                        sent = sentiment_results[i] if i < len(sentiment_results) else {}
                        aspects = aspect_results[i] if i < len(aspect_results) else {}
                        
                        sentiment_label = sent.get('labels', [None])[0]
                        sentiment_score = sent.get('scores', [0])[0] if sent.get('scores') else 0
                        stars = sentiment_to_stars(sentiment_label, sentiment_score) if sentiment_label else 0
                        
                        row = {
                            'Review': review[:80] + '...' if len(review) > 80 else review,
                            'Sentiment': sentiment_label or 'N/A',
                            'Confidence': f"{sentiment_score:.1%}",
                            'Stars': stars
                        }
                        
                        if aspects.get('labels'):
                            for j, (label, score) in enumerate(zip(aspects['labels'][:2], aspects['scores'][:2])):
                                row[f'Aspect {j+1}'] = f"{label} ({score:.1%})"
                        
                        batch_results.append(row)
                    
                    result_df = pd.DataFrame(batch_results)
                    
                    st.success("✅ Batch complete!")
                    st.markdown("---")
                    st.subheader(f"📈 Results ({len(batch_results)} reviews)")
                    
                    # Charts
                    col_pie, col_bar = st.columns([1, 1])
                    
                    sentiment_counts = result_df['Sentiment'].value_counts()
                    with col_pie:
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    star_counts = result_df['Stars'].value_counts().sort_index()
                    with col_bar:
                        fig_bar = px.bar(
                            x=star_counts.index,
                            y=star_counts.values,
                            labels={'x': 'Rating', 'y': 'Count'},
                            title="Star Rating Distribution",
                            color=star_counts.index,
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.write("**Detailed Results:**")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    # Download
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download as CSV",
                        data=csv_data,
                        file_name="batch_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Batch processing failed: {e}")

# ===== TAB 3: ABOUT & HELP =====
with tab3:
    st.markdown("### ℹ️ About PulseLens")
    
    st.markdown("""
    **PulseLens** analyzes customer feedback using AI-powered zero-shot classification.
    
    #### How It Works
    - Uses Hugging Face transformer model for sentiment & aspect classification
    - **Sentiment:** Classifies reviews as positive/neutral/negative
    - **Aspects:** Scores how strongly specific topics (e.g., "service", "price") appear in reviews
    - **Star Rating:** Maps sentiment + confidence to 1-5 stars
    
    #### Features
    ✅ Single review analysis with instant results  
    ✅ Batch CSV processing (100+ reviews at once)  
    ✅ Industry presets (Restaurant, Electronics, Fashion, etc.)  
    ✅ Custom aspect selection  
    ✅ Export results as CSV  
    
    #### Star Rating Logic
    | Sentiment | Confidence | Rating |
    |-----------|-----------|--------|
    | Positive | ≥90% | ⭐⭐⭐⭐⭐ |
    | Positive | 75-90% | ⭐⭐⭐⭐ |
    | Positive | <75% | ⭐⭐⭐ |
    | Neutral | Any | ⭐⭐⭐ |
    | Negative | 60-85% | ⭐⭐ |
    | Negative | ≥85% | ⭐ |
    
    #### Industry Aspect Presets
    """)
    
    for industry, aspects in GROUPED_ASPECTS.items():
        st.markdown(f"**{industry}**  {', '.join(aspects)}")
    
    st.markdown("""
    #### FAQ
    
    **Q: How fast is analysis?**  
    A: First run downloads model (~2 min), then <1s per review.
    
    **Q: Can I use custom aspects?**  
    A: Yes, just type them. No retraining needed.
    
    **Q: Is data stored?**  
    A: No, session-only. Results disappear when you close the page.
    
    **Q: CSV requirements?**  
    A: Must have column named `review`.
    """)

st.markdown('</div>', unsafe_allow_html=True)
