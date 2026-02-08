# PulseLens — Customer Feedback Sentiment & Aspect Analyzer

PulseLens is a lightweight, easy-to-deploy Streamlit application that uses Hugging Face zero-shot classification to extract sentiment and user-defined aspects from customer reviews. Perfect for quick experimentation, team analytics, and feedback analysis workflows.

## Key Features

✅ **Zero-shot sentiment classification** (positive / neutral / negative)  
✅ **Aspect relevance scoring** for custom aspect lists  
✅ **Single-review interactive mode** with instant results  
✅ **CSV batch processing** - upload multiple reviews at once  
✅ **Industry presets** - click-to-add aspects for 6 industries  
✅ **Charts & visualizations** - see results in pie/bar charts  
✅ **One-click deployment** - deploy to Streamlit Cloud for free  
✅ **No backend required** - runs entirely on Streamlit  

## Quick Start (Local)

### Prerequisites
- Python 3.10+ 
- Virtual environment (recommended)

### 1. Clone & Setup

```bash
git clone https://github.com/scmlewis/PulseLens.git
cd PulseLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: First run will download the Hugging Face model (~4GB) - this takes a few minutes.

### 3. Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser - that's it! 🎉

---

## Deploy to Streamlit Cloud (Free)

The easiest way to share your app with others:

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Streamlit Cloud"
git push origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch, and `app.py` as main file
5. Click "Deploy"

**That's it!** Your app is now live at `share.streamlit.io/your-username/your-repo/main`

---

## How to Use

### **Tab 1: Single Review Analysis**
1. Paste or type a customer review
2. Select an **industry preset** (automatic aspect recommendation) OR manually choose specific aspects
3. Click **Classify Now**
4. See:
   - 😊/😐/😞 **Sentiment** (positive/neutral/negative)
   - ⭐ **Star Rating** (1-5 stars mapped from sentiment confidence)
   - **Aspect Relevance Scores** (how strongly each aspect appears in the review)

### **Tab 2: Batch Reviews**
1. **Upload CSV** with a `review` column, OR **paste reviews** (one per line)
2. Select aspects to analyze
3. Click **Classify Batch**
4. Get:
   - Results table with all analyses
   - Sentiment distribution pie chart
   - Star rating bar chart
   - Download button for CSV export

### **Tab 3: About & Help**
- Industry aspect suggestions (click to add to your selection)
- Explanation of sentiment vs. rating
- FAQ & tips

---

## Project Structure

```
.
├── app.py                          # Main Streamlit app ✨
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit Cloud config
├── static/
│   └── styles.css                 # Dark theme styling
├── tests/
│   ├── test_sentiment.py          # Unit tests
│   └── __pycache__/
├── run_tests.py                   # Test runner
├── README.md
└── .gitignore
```

---

## Technology

| Component | Tool |
|-----------|------|
| **Framework** | Streamlit 1.37+ |
| **NLP Model** | Hugging Face `facebook/bart-large-mnli` (zero-shot classification) |
| **ML Backend** | PyTorch + Transformers |
| **Data** | Pandas |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Cloud (free tier) |

---

## Understanding the Results

### **Sentiment**
The AI classifies the overall tone as:
- **😊 Positive** - expresses satisfaction and approval
- **😐 Neutral** - balanced or factual opinion  
- **😞 Negative** - expresses dissatisfaction or criticism

### **Confidence Score**
How sure the AI is about its prediction (0-100%). Higher = more reliable.

### **Star Rating Mapping**
Stars are automatically mapped from sentiment + confidence:
- **5 stars** ⭐⭐⭐⭐⭐ - Positive with high confidence (≥90%)
- **4 stars** ⭐⭐⭐⭐ - Positive with medium confidence (≥75%)
- **3 stars** ⭐⭐⭐ - Neutral (all neutral reviews)
- **2 stars** ⭐⭐ - Negative with medium confidence (≥60%)
- **1 star** ⭐ - Negative with high confidence (≥85%)

### **Aspect Relevance Score**
For each aspect you select (e.g., "service", "price", "quality"), the AI scores how strongly that aspect appears in the review (0.0 - 1.0):
- **0.8-1.0** - Highly relevant, strongly mentioned
- **0.5-0.8** - Moderately relevant
- **0.0-0.5** - Weakly relevant or not mentioned

---

## Aspect Presets (Industries)

### 🍽️ **Restaurant**
food, service, ambience, price, delivery, staff, product quality

### 💻 **Electronics**
battery, display, camera, performance, durability, shipping, support

### 👗 **Fashion**
fit, material, style, comfort, design, price

### 🛒 **Supermarket**
freshness, variety, checkout, customer service, packaging, speed

### 📚 **Books**
plot, characters, writing, pacing, ending, value

### 🏨 **Hotel**
cleanliness, location, amenities, room, wifi, maintenance

---

## FAQ

**Q: Can I use custom aspects?**  
A: Yes! Just type them in the multiselect box or click industry chips to get started.

**Q: What if my CSV has a different column name?**  
A: The app looks for a column named exactly `review`. Rename your column or upload a new file.

**Q: How long does the first run take?**  
A: The first time you analyze a review, it downloads the AI model (~4GB). This takes 3-5 minutes. After that, analysis is instant.

**Q: Can I work offline?**  
A: Yes - once the model is downloaded, the app works completely offline.

**Q: Is there a limit on review length?**  
A: No, the model handles short tweets to long paragraphs equally well.

**Q: Can I export batch results?**  
A: Yes! Click the **Download Results as CSV** button after running batch analysis.

---

## Development & Testing

### Run Unit Tests
```bash
python run_tests.py
```

Tests verify the sentiment-to-stars conversion logic.

### Add New Tests
Edit `tests/test_sentiment.py` and add your test function:
```python
def test_my_case():
    assert sentiment_to_stars('positive', 0.92) == 5
```

---

## Limitations & Future Ideas

- ⏳ **Session-only** - results don't persist across app restarts (by design)
- 🤖 **Single model** - uses `facebook/bart-large-mnli`, no model switching yet
- 💾 **No data export** - can save CSV from batch mode
- 🔗 **No API** - frontend-only, no programmatic access
- 📱 **UI-only responsive** - optimized for desktop

### Ideas for v2:
- Database storage for result history
- User authentication & saving favorite presets
- REST API for programmatic access
- Fine-tuned models for specific domains
- Comparison & trend analysis across multiple batches
- Export to PDF with formatted reports

---

## License

[Add LICENSE file - currently unlicensed]

---

## Questions or Issues?

- 🐛 Found a bug? [Open an issue](https://github.com/scmlewis/PulseLens/issues)
- 💬 Have feedback? [Start a discussion](https://github.com/scmlewis/PulseLens/discussions)
- 📧 Email: [your email here]

---

## Changelog

**v2.0** (Feb 2026)
- ✨ Streamlined for single-click Streamlit Cloud deployment
- 🎨 Modern dark theme with Inter font
- ⚡ Instant local analysis (no backend)
- 📊 Batch processing with charts
- 🏭 Industry aspect presets

**v1.0** - Original version

- Add CSV/DB persistence for batch results and historical snapshots.
- Add a `Dockerfile` for reproducible builds and to avoid local-native compile issues.
- Add caching of text→classification to reduce repeated inference costs.

## Contributing

- Open an issue for feature requests or bugs.
- For code contributions, fork the repo, create a feature branch, and open a PR with a clear description of changes. Add tests for new logic.

## License

This repository does not include a license file. Add `LICENSE` (for example MIT) if you want to make the project open-source.

## Contact

For questions or help, reach out to the repository owner.
