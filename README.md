# PulseLens — Customer Feedback Sentiment & Aspect Analyzer

PulseLens (previously *Customer Feedback Sentiment & Aspect Classifier*) is a lightweight Streamlit application that uses Hugging Face zero-shot classification to extract sentiment and user-defined aspects from free-text customer reviews. It's designed for quick experimentation, small-team analytics, and human-in-the-loop labeling workflows.

## Features

- Zero-shot sentiment classification (positive / neutral / negative)
- Aspect relevance scoring for custom aspect lists
- Single-review interactive mode and CSV batch processing
- Click-to-add aspect presets by industry
- Chunked batch processing with progress UI

## Quick Links

- App entry: `app.py`
- Tests: `tests/test_sentiment.py`, runner: `run_tests.py`
- Styles: `static/styles.css`

## Getting started (local)

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

> Notes
>
>- If you encounter issues building native packages (e.g., `numpy`), use a Python version with prebuilt wheels (3.10/3.11) or install the Visual Studio Build Tools on Windows. Alternatively use a Docker container for reproducible execution.
>- The app uses the HF model `facebook/bart-large-mnli` by default; this can be heavy — consider swapping to a smaller model for low-memory environments.

## Usage

- **Single Review**: Paste a review, choose aspects (or an industry preset), and click *Classify Now* to see sentiment, star mapping, and aspect relevance scores.
- **Batch Reviews**: Upload a CSV with a `review` column or paste reviews (one per line). Choose aspects and click *Classify Batch* to produce a results table and charts.

## Development & Testing

- Run unit tests:

```powershell
python run_tests.py
```

- Add or customize linters/formatters (e.g. `black`, `ruff`) and consider adding `pre-commit` hooks.

## Suggested Improvements

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
