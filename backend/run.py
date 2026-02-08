#!/usr/bin/env python
"""
PulseLens Flask Backend Startup Script
Initializes the classifier and runs the Flask server.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("PulseLens - Flask Backend Server")
print("=" * 60)
print()

# Initialize the classifier first
print("🔄 Initializing Hugging Face model (facebook/bart-large-mnli)...")
print("   This may take a minute on first run.")
print()

try:
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Please check your transformers/torch installation.")
    sys.exit(1)

print()

# Import Flask app
try:
    from backend.app import app, set_classifier
    from backend.storage import ensure_dirs
    
    # Pass classifier to Flask app
    set_classifier(classifier)
    
    # Ensure data directories exist
    ensure_dirs()
    print("✅ Flask app initialized")
except Exception as e:
    print(f"❌ Error initializing Flask app: {e}")
    sys.exit(1)

print()
print("-" * 60)
print("🚀 Starting Flask server...")
print("-" * 60)
print()
print("Backend running at: http://127.0.0.1:5000")
print("Health check:       GET  http://127.0.0.1:5000/health")
print()
print("Available endpoints:")
print("  - POST   /analyze          Analyze a single review")
print("  - POST   /analyze/batch    Analyze multiple reviews")
print("  - GET    /history          Get analysis history")
print("  - GET    /insights         Get aggregated insights")
print("  - GET    /export/csv       Export results as CSV")
print()
print("Press Ctrl+C to stop the server")
print()

try:
    app.run(debug=False, port=5000, host="127.0.0.1", use_reloader=False)
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user")
    sys.exit(0)
