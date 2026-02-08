# Customer Feedback Sentiment & Aspect Classifier

This repository contains a Streamlit app for sentiment and aspect classification of customer feedback using a zero-shot transformer model.

## Run locally

Install dependencies and run Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If you have limited RAM or no GPU, consider using a smaller zero-shot model or running with smaller batch sizes.

## GitHub & CI

This repository is prepared for GitHub. A lightweight GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs the unit tests with `pytest` on pushes and pull requests to `main`.

To publish this repository to GitHub and connect Streamlit Cloud:

```powershell
cd customer_feedback_analyzer
git remote add origin https://github.com/<your-org-or-user>/<repo-name>.git
git branch -M main
git push -u origin main
```

Then create a new app on Streamlit Cloud and point it to the GitHub repo and branch.
