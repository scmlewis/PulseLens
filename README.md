title: Customer Feedback Sentiment & Aspect Classifier
emoji: 📊
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
short_description: Analyze customer feedback with categorization

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
## Run locally

Install dependencies and run Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If you have limited RAM or no GPU, consider using a smaller zero-shot model or run with fewer batch sizes.