"""
PulseLens - Customer Feedback Analyzer
Lightweight Streamlit app - single deploy, no backend required.
Session-based results storage (works with Streamlit Cloud).
"""
import os
import streamlit as st
import pandas as pd
import random
import json
import uuid
from datetime import datetime
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

# ==================== RESULTS STORAGE (Session-based for Streamlit Cloud) ====================

def init_results_storage():
    """Initialize results storage in session state."""
    if 'results_list' not in st.session_state:
        st.session_state['results_list'] = []

def save_result(review_text: str, sentiment: str, confidence: float, stars: int, 
                industry: str = "", aspects: dict = None, notes: str = "") -> str:
    """
    Save a single review analysis result to session state.
    Returns the result ID.
    """
    result_id = str(uuid.uuid4())[:8]
    result = {
        "id": result_id,
        "timestamp": datetime.now().isoformat(),
        "review_text": review_text,
        "sentiment": sentiment,
        "confidence": float(confidence),
        "stars": int(stars),
        "industry": industry,
        "aspects": aspects or {},
        "favorited": False,
        "notes": notes
    }
    
    # Add to session state (stored in memory for this session)
    st.session_state['results_list'].insert(0, result)  # Insert at beginning for newest first
    
    return result_id

def load_all_results() -> list:
    """Load all analysis results from session state."""
    init_results_storage()
    return st.session_state.get('results_list', [])

def delete_result(result_id: str):
    """Delete a result by ID from session state."""
    init_results_storage()
    st.session_state['results_list'] = [
        r for r in st.session_state['results_list'] 
        if r.get("id") != result_id
    ]
    return True

def update_result_favorite(result_id: str, favorited: bool):
    """Toggle favorite status of a result in session state."""
    init_results_storage()
    for result in st.session_state['results_list']:
        if result.get("id") == result_id:
            result["favorited"] = favorited
            return True
    return False

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

# Initialize session state for results storage
if 'results_list' not in st.session_state:
    st.session_state['results_list'] = []

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
tab1, tab2, tab3 = st.tabs(["📝 Analyze One Review", "📊 Analyze Multiple Reviews", "📈 My Results"])

# ===== TAB 1: SINGLE REVIEW ANALYSIS =====
with tab1:
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    if 'aspects_select' not in st.session_state:
        st.session_state['aspects_select'] = []
    if 'industry_select' not in st.session_state:
        st.session_state['industry_select'] = ''
    
    st.markdown("## 💬 Analyze a Single Review")
    st.markdown("Paste or type a customer review below to get instant sentiment analysis and aspect scores.", unsafe_allow_html=True)
    
    # Industry selection (OUTSIDE form - callbacks not allowed inside forms)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**🏭 Industry (Optional)**")
        st.markdown("<small>Auto-populate aspects when you select an industry</small>", unsafe_allow_html=True)
        industries = ["-- Select industry --"] + list(GROUPED_ASPECTS.keys())
        st.selectbox(
            "Choose an industry:",
            industries,
            key='industry_select',
            label_visibility='collapsed',
            on_change=on_industry_select
        )
    
    with col2:
        st.markdown("**✨ Quick Actions**")
        st.markdown("<small>Try a sample review or clear the field</small>", unsafe_allow_html=True)
        col_sample, col_clear = st.columns(2)
        with col_sample:
            st.button("Load Sample", on_click=set_sample, key="gen_sample_btn", use_container_width=True)
        with col_clear:
            st.button("Clear", on_click=clear_text, key="clear_btn", use_container_width=True)
    
    st.markdown("---")
    
    # Form for main inputs (INSIDE form - no callbacks allowed)
    with st.form(key='single_review_form'):
        st.markdown("**📝 Review Text**")
        st.markdown("<small>Enter a customer review (at least 10 characters)</small>", unsafe_allow_html=True)
        text = st.text_area(
            "Type or paste a review:",
            height=130,
            key="review_text",
            label_visibility="collapsed",
            placeholder="e.g., 'The service was great, but the food was cold...'"
        )
        
        st.markdown("**Aspects to Analyze**")
        st.markdown("<small>Select 1 or more aspects (or use industry presets above)</small>", unsafe_allow_html=True)
        st.multiselect(
            "Choose aspects:",
            options=get_all_aspects(),
            default=st.session_state.get('aspects_select', []),
            key='aspects_select',
            label_visibility='collapsed'
        )
        
        submit = st.form_submit_button(label="🔍 Analyze Now", type="primary", use_container_width=True)

    if submit:
        if not st.session_state.get('review_text', '').strip():
            st.error("❌ Please enter a review.")
        elif len(st.session_state.get('review_text', '').strip()) < 10:
            st.error("❌ Review must be at least 10 characters long.")
        elif not st.session_state.get('aspects_select'):
            st.error("❌ Please select at least one aspect.")
        else:
            with st.spinner("🔄 Analyzing sentiment and aspects... This may take a moment."):
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
                    
                    sentiment = sentiment_result['labels'][0]
                    score = sentiment_result['scores'][0]
                    stars = sentiment_to_stars(sentiment, score)
                    
                    # Save result
                    save_result(
                        review_text=st.session_state.get('review_text'),
                        sentiment=sentiment,
                        confidence=score,
                        stars=stars,
                        industry=st.session_state.get('industry_select', ''),
                        aspects=dict(zip(aspect_result['labels'], aspect_result['scores']))
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.success("✅ Analysis complete!")
                    st.subheader("📊 Results")
                    
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
                            color_continuous_scale='Blues',
                            title="Aspect Scores"
                        )
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                        fig.update_xaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_table:
                        display_df = aspect_df.copy()
                        display_df.columns = ['Aspect', 'Relevance Score']
                        display_df['Relevance Score'] = display_df['Relevance Score'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("Try refreshing the page or checking your review text.")

# ===== TAB 2: BATCH REVIEWS =====
with tab2:
    st.markdown("## 📊 Analyze Multiple Reviews")
    st.markdown("Upload a CSV file or paste multiple reviews to analyze them in bulk.", unsafe_allow_html=True)
    
    if 'batch_aspects_select' not in st.session_state:
        st.session_state['batch_aspects_select'] = []
    
    if 'use_paste' not in st.session_state:
        st.session_state['use_paste'] = False
    
    input_method = st.radio(
        "Choose input method:",
        ["📤 Upload CSV", "📋 Paste Reviews"],
        horizontal=True,
        key="input_method"
    )
    
    st.markdown("---")
    
    with st.form(key='batch_form'):
        if input_method == "📤 Upload CSV":
            st.markdown("**Requirements:** CSV file with a column named `review`")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="batch_file")
            batch_input = uploaded_file
        else:
            st.markdown("**Paste reviews** one per line (minimum 2 characters each)")
            pasted_text = st.text_area(
                "Paste reviews:",
                height=150,
                label_visibility='collapsed',
                key="batch_manual",
                placeholder="e.g., Great service!\nFood was cold.\nWill come back."
            )
            batch_input = pasted_text
        
        st.markdown("---")
        st.markdown("**Select Aspects to Analyze**")
        st.markdown("<small>Choose at least one aspect</small>", unsafe_allow_html=True)
        st.multiselect(
            "Aspects:",
            options=get_all_aspects(),
            default=st.session_state.get('batch_aspects_select', []),
            key='batch_aspects_select',
            label_visibility='collapsed'
        )
        
        submit_batch = st.form_submit_button("⚙️ Analyze Batch", type="primary", use_container_width=True)
    
    if submit_batch:
        reviews_list = []
        
        if input_method == "📤 Upload CSV":
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'review' not in df.columns:
                        st.error("❌ CSV must have a column named `review`. Found columns: " + ", ".join(df.columns))
                    else:
                        reviews_list = df['review'].dropna().astype(str).tolist()
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
            else:
                st.error("❌ Please upload a CSV file.")
        else:
            if pasted_text and pasted_text.strip():
                reviews_list = [r.strip() for r in pasted_text.split('\n') if r.strip() and len(r.strip()) >= 2]
            else:
                st.error("❌ Please paste at least one review.")
        
        if not reviews_list:
            st.error("❌ No valid reviews found.")
        elif not st.session_state.get('batch_aspects_select'):
            st.error("❌ Please select at least one aspect.")
        else:
            with st.spinner(f"🔄 Analyzing {len(reviews_list)} review(s)... This may take a moment."):
                try:
                    # Sentiment analysis
                    progress_text = "Analyzing sentiment..."
                    progress_bar = st.progress(0, text=progress_text)
                    sentiment_results = classify_batch(
                        reviews_list,
                        SENTIMENT_LABELS,
                        get_classifier,
                        chunk_size=32
                    )
                    progress_bar.empty()
                    
                    # Aspect analysis
                    progress_text = "Analyzing aspects..."
                    progress_bar = st.progress(0, text=progress_text)
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
                            'Review': review[:75] + '...' if len(review) > 75 else review,
                            'Sentiment': sentiment_label or 'N/A',
                            'Confidence': f"{sentiment_score:.1%}",
                            'Stars': stars
                        }
                        
                        if aspects.get('labels'):
                            for j, (label, score) in enumerate(zip(aspects['labels'][:2], aspects['scores'][:2])):
                                row[f'Aspect {j+1}'] = f"{label} ({score:.1%})"
                        
                        batch_results.append(row)
                    
                    result_df = pd.DataFrame(batch_results)
                    
                    st.success(f"✅ Analysis complete! Processed {len(batch_results)} reviews.")
                    st.markdown("---")
                    st.subheader(f"📈 Results Summary")
                    
                    # Charts
                    col_pie, col_bar = st.columns([1, 1])
                    
                    sentiment_counts = result_df['Sentiment'].value_counts()
                    with col_pie:
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={
                                'positive': '#22c55e',
                                'neutral': '#f59e0b',
                                'negative': '#ef4444'
                            }
                        )
                        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    star_counts = result_df['Stars'].value_counts().sort_index()
                    with col_bar:
                        fig_bar = px.bar(
                            x=star_counts.index,
                            y=star_counts.values,
                            labels={'x': 'Rating', 'y': 'Count'},
                            title="Star Rating Distribution",
                            color_discrete_sequence=['#4F8BF9'] * len(star_counts)
                        )
                        fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("---")
                    st.write("**Detailed Results:**")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    # Download
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")
                    st.info("Try with fewer reviews or check your input format.")

# ===== TAB 3: MY RESULTS =====
with tab3:
    st.markdown("## 📈 My Results")
    st.markdown("View, search, filter, and manage your analysis results during this session.", unsafe_allow_html=True)
    
    st.info("💡 **Note:** Results are stored in your current session only. They will be cleared when you refresh the page or the app restarts. To save results, export them as CSV.", icon="ℹ️")
    
    all_results = load_all_results()
    
    if not all_results:
        st.info("📭 No results yet. Analyze some reviews in the tabs above to see them here!", icon="ℹ️")
    else:
        st.success(f"📊 Total results: {len(all_results)}")
        
        # Filters
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            search_text = st.text_input("🔍 Search reviews:", placeholder="Search by review text...")
        
        with col2:
            filter_sentiment = st.multiselect(
                "Sentiment:",
                ["positive", "neutral", "negative"],
                default=["positive", "neutral", "negative"],
                key="filter_sentiment"
            )
        
        with col3:
            filter_stars = st.slider("Minimum stars:", 1, 5, 1, key="filter_stars")
        
        # Apply filters
        filtered_results = []
        for result in all_results:
            # Search filter
            if search_text and search_text.lower() not in result.get("review_text", "").lower():
                continue
            # Sentiment filter
            if result.get("sentiment") not in filter_sentiment:
                continue
            # Stars filter
            if result.get("stars", 0) < filter_stars:
                continue
            filtered_results.append(result)
        
        st.markdown("---")
        
        if not filtered_results:
            st.warning("No results match your filters.")
        else:
            st.subheader(f"Showing {len(filtered_results)} result(s)")
            
            # Results display
            for i, result in enumerate(filtered_results):
                with st.expander(
                    f"{'⭐' * result['stars']} {result['sentiment'].title()} - {result['review_text'][:60]}...",
                    expanded=(i == 0)
                ):
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**Review:** {result['review_text']}")
                        st.write(f"**Timestamp:** {result['timestamp'][:10]} {result['timestamp'][11:16]}")
                    
                    with col2:
                        st.write(f"**Sentiment:** {result['sentiment'].title()}")
                        st.write(f"**Confidence:** {result['confidence']:.1%}")
                        st.write(f"**Stars:** {'⭐' * result['stars']}")
                    
                    with col3:
                        favorite_state = result.get("favorited", False)
                        if st.button(f"{'❤️' if favorite_state else '🤍'}", key=f"fav_{result['id']}"):
                            update_result_favorite(result['id'], not favorite_state)
                            st.rerun()
                    
                    with col4:
                        if st.button("🗑️", key=f"del_{result['id']}"):
                            delete_result(result['id'])
                            st.rerun()
                    
                    # Aspects
                    if result.get('aspects'):
                        st.write("**Aspects:**")
                        aspects_df = pd.DataFrame({
                            'Aspect': list(result['aspects'].keys()),
                            'Score': list(result['aspects'].values())
                        }).sort_values('Score', ascending=False)
                        
                        aspects_df['Score'] = aspects_df['Score'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(aspects_df, use_container_width=True, hide_index=True)
            
            # Bulk export
            st.markdown("---")
            export_df = pd.DataFrame([
                {
                    'Review': r['review_text'][:100],
                    'Sentiment': r['sentiment'],
                    'Confidence': f"{r['confidence']:.1%}",
                    'Stars': r['stars'],
                    'Timestamp': r['timestamp'][:10],
                    'Favorited': '❤️' if r.get('favorited') else ''
                }
                for r in filtered_results
            ])
            
            csv_export = export_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Export Filtered Results as CSV",
                data=csv_export,
                file_name=f"results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown('</div>', unsafe_allow_html=True)
