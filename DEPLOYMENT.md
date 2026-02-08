# 🚀 Deployment Guide - PulseLens

Deploy your PulseLens app to Streamlit Cloud in 5 minutes!

## Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- This repo pushed to GitHub

## Step 1: Prepare GitHub Repository

```bash
# From project root
git add .
git commit -m "feat: refactor for Streamlit Cloud deployment with session-based storage"
git push origin main
```

## Step 2: Deploy to Streamlit Cloud

### Via Streamlit Cloud Dashboard

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select:
   - **Repository:** `scmlewis/customer-feedback` (or your repo)
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `app.py`
4. Click **"Deploy"**

Streamlit will install dependencies from `requirements.txt` and start the app. First deployment takes ~2-5 minutes.

### Via Streamlit CLI (Alternative)

```bash
streamlit run app.py --deploy
```

## Features Enabled

✅ Full dark theme with refined CSS  
✅ Single review analysis  
✅ Batch CSV processing  
✅ Session-based results (search, filter, favorite, delete, export)  
✅ Responsive design (mobile, tablet, desktop)  
✅ All sentiment analysis and aspect scoring  

## Important: Session-Based Storage

**Results are NOT persistent:**
- Results stored in `st.session_state` (in-memory)
- Cleared when:
  - User refreshes the page
  - App restarts
  - Streamlit Cloud restarts the instance

**To save results:** Users must export as CSV from the "My Results" tab.

**Workaround if you need persistence:**
See [PERSISTENCE.md](./PERSISTENCE.md) for options to add a database.

## Testing Locally First

Before deploying, test locally:

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Visit `http://localhost:8501` and test all features.

## Performance Notes

- **Model Loading:** First access (any user) takes 2-5 minutes to download the HF model (~4GB)
  - Subsequent accesses are fast (<1s per review)
  - Model cached in Streamlit Cloud resources
- **Batch Processing:** 100 reviews take ~1-2 minutes
- **Memory:** Limited to ~1GB per session on free tier

## Troubleshooting

### App won't start
- Check `requirements.txt` is in repo root
- Verify `app.py` runs locally first
- Check Streamlit Cloud logs (click app → "Manage app")

### Model download fails
- Model is large (~4GB) and may timeout
- Upgrade to Streamlit+ for more resources, or
- Use smaller HF model (e.g., `distilbert` instead of `bart`)

### Results disappearing
- This is expected! Results are session-only by design
- User must export to CSV to keep results

### Slow performance
- First load is slow (model download)
- Batch processing only runs one at a time
- Consider upgrading or limiting to smaller batches

## Customization

### Change Model
Edit `app.py` line ~150:
```python
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

Swap model names:
- `distilbert-base-uncased`: Smaller, faster (~300MB)
- `roberta-large-mnli`: Larger, more accurate (~500MB)
- `bart-large-mnli`: Current choice, balanced (~1.6GB)

### Monitor App
- Check logs: Click app → "Manage app" → "View logs"
- Check usage: "Manage app" → "App analytics"

## Next Steps

**For persistent results**, see [PERSISTENCE.md](./PERSISTENCE.md) to add:
- Firebase Realtime DB (free tier available)
- Supabase PostgreSQL (free tier available)
- Google Sheets (free)

---

**Questions?** Check Streamlit docs: https://docs.streamlit.io/
