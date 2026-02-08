# 💾 Adding Persistent Storage to PulseLens

Currently, results are stored **in-session only**. Here are options to add persistent storage.

## Option 1: Supabase (PostgreSQL) - Recommended 🌟

**Free tier:** 500MB database, 2GB file storage, perfect for this use case.

### Setup (5 minutes)

1. **Create Supabase account** → https://supabase.com
2. **Create new project** (choose region near you)
3. **Create table** with SQL editor:

```sql
CREATE TABLE results (
  id TEXT PRIMARY KEY,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  review_text TEXT NOT NULL,
  sentiment TEXT,
  confidence FLOAT,
  stars INT,
  industry TEXT,
  aspects JSONB,
  favorited BOOLEAN DEFAULT FALSE,
  notes TEXT
);

CREATE INDEX idx_sentiment ON results(sentiment);
CREATE INDEX idx_timestamp ON results(timestamp DESC);
```

4. **Get credentials:**
   - Go to Settings → API → Copy `SUPABASE_URL` and `SUPABASE_KEY`
   - Add to Streamlit Cloud Secrets (see below)

5. **Install package:**
```bash
pip install supabase
```

6. **Update `app.py`:**

```python
from supabase import create_client
import streamlit as st

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

def save_result_db(review_text, sentiment, confidence, stars, industry="", aspects=None):
    supabase = get_supabase()
    import uuid
    result = {
        "id": str(uuid.uuid4())[:8],
        "review_text": review_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "stars": stars,
        "industry": industry,
        "aspects": aspects or {},
    }
    supabase.table("results").insert(result).execute()
    return result["id"]

def load_results_db():
    supabase = get_supabase()
    return supabase.table("results").select("*").order("timestamp", desc=True).execute().data
```

7. **Add secrets to Streamlit Cloud:**
   - Go to app dashboard → "Manage app" → "Secrets"
   - Add:
     ```
     [SUPABASE_URL]
     SUPABASE_KEY
     ```

### Pros/Cons
- ✅ Free, easy to set up
- ✅ Scalable (PostgreSQL)
- ✅ Works on Streamlit Cloud
- ❌ Requires Internet connection

---

## Option 2: Firebase Realtime Database

**Free tier:** 100 simultaneous connections, 1GB storage.

### Setup (5 minutes)

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create project
3. Enable Realtime Database (Blaze plan - pay-as-you-go, no cost at low use)
4. Create service account key (Settings → Service Accounts → Generate)
5. Download JSON file

```bash
pip install firebase-admin
```

```python
import firebase_admin
from firebase_admin import db
import json
import streamlit as st

@st.cache_resource
def init_firebase():
    creds = json.loads(st.secrets["FIREBASE_KEY"])
    firebase_admin.initialize_app(options={
        "databaseURL": st.secrets["FIREBASE_URL"]
    }, credentials=firebase_admin.credentials.Certificate(creds))

def save_result_db(review_text, sentiment, confidence, stars, industry="", aspects=None):
    db.reference("results").push({
        "review_text": review_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "stars": stars,
        "industry": industry,
        "aspects": aspects or {},
        "timestamp": datetime.now().isoformat()
    })
```

### Pros/Cons
- ✅ Real-time sync
- ✅ Free tier decent
- ✅ Easy to set up
- ❌ Less queryable than SQL
- ❌ Can trigger billing if usage spikes

---

## Option 3: Google Sheets - Simplest

**Free tier:** Unlimited (uses your Google Drive).

```bash
pip install gspread oauth2client
```

```python
import gspread
import streamlit as st

@st.cache_resource
def get_sheets():
    creds = json.loads(st.secrets["GOOGLE_SHEETS_KEY"])
    return gspread.service_account_from_dict(creds)

def save_result_sheets(review_text, sentiment, confidence, stars):
    gc = get_sheets()
    sheet = gc.open("PulseLens Results").worksheet(0)
    sheet.append_row([
        datetime.now().isoformat(),
        review_text,
        sentiment,
        f"{confidence:.1%}",
        stars
    ])
```

### Pros/Cons
- ✅ Simplest to set up
- ✅ Free
- ✅ Easy to share/analyze results
- ❌ Slower (API calls)
- ❌ Limited to ~2000 rows before slow

---

## Recommendation

**For PulseLens, use Supabase:**
- Good balance of free tier + features
- Proper database = fast queries
- Easy to add export/analytics later
- Works perfectly on Streamlit Cloud

---

## Migration Path

1. Deploy current app (in-session)
2. Test and validate UX
3. When ready, add database option:
   - Keep session storage as fallback
   - Sync to database in background
   - Let users toggle "Save to cloud"
