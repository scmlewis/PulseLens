"""
Flask backend for PulseLens customer feedback analyzer.
Handles data persistence, search, and insights computation.
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import uuid
import io
import os

from .models import AnalysisResult, sentiment_to_stars, classify_sentiment, classify_aspects, GROUPED_ASPECTS
from .storage import (
    save_analysis, load_all_analyses, search_analyses, 
    compute_insights, save_insights_cache, load_insights_cache,
    save_batch, load_batch, export_csv, delete_analysis
)

app = Flask(__name__)
CORS(app)

# Global classifier instance (will be loaded from Streamlit)
classifier = None


def set_classifier(clf):
    """Set the classifier instance from Streamlit."""
    global classifier
    classifier = clf


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/analyze", methods=["POST"])
def analyze_single():
    """
    Analyze a single review.
    
    Request JSON:
    {
        "review_text": "...",
        "aspects": ["aspect1", "aspect2", ...],
        "title": "..."  # optional
    }
    """
    try:
        data = request.get_json()
        review_text = data.get("review_text", "").strip()
        aspects = data.get("aspects", [])
        title = data.get("title", "")
        
        if not review_text:
            return jsonify({"error": "review_text is required"}), 400
        
        if not aspects:
            return jsonify({"error": "aspects list is required"}), 400
        
        # Classify sentiment
        sentiment, confidence = classify_sentiment(review_text, classifier)
        stars = sentiment_to_stars(sentiment, confidence)
        
        # Classify aspects
        aspect_scores = classify_aspects(review_text, aspects, classifier)
        
        # Create result
        result = AnalysisResult(
            review_text=review_text,
            sentiment=sentiment,
            confidence=confidence,
            stars=stars,
            aspects=aspects,
            aspect_scores=aspect_scores,
            created_at=datetime.now().isoformat(),
            title=title or f"Review_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{sentiment}",
            analysis_id=str(uuid.uuid4())
        )
        
        # Save to local storage
        save_analysis(result)
        
        return jsonify({
            "success": True,
            "analysis": result.to_dict()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """
    Analyze a batch of reviews.
    
    Request JSON:
    {
        "reviews": ["review1", "review2", ...],
        "aspects": ["aspect1", "aspect2", ...],
        "batch_name": "..."
    }
    """
    try:
        data = request.get_json()
        reviews = data.get("reviews", [])
        aspects = data.get("aspects", [])
        batch_name = data.get("batch_name", "Batch")
        
        if not reviews:
            return jsonify({"error": "reviews list is required"}), 400
        
        if not aspects:
            return jsonify({"error": "aspects list is required"}), 400
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        results = []
        
        # Analyze each review
        for i, review_text in enumerate(reviews):
            if not review_text or not review_text.strip():
                continue
            
            sentiment, confidence = classify_sentiment(review_text, classifier)
            stars = sentiment_to_stars(sentiment, confidence)
            aspect_scores = classify_aspects(review_text, aspects, classifier)
            
            result = AnalysisResult(
                review_text=review_text,
                sentiment=sentiment,
                confidence=confidence,
                stars=stars,
                aspects=aspects,
                aspect_scores=aspect_scores,
                created_at=datetime.now().isoformat(),
                analysis_id=f"{batch_id}_{i}"
            )
            results.append(result)
        
        # Save batch results
        save_batch(batch_id, batch_name, aspects, results)
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "num_reviews_analyzed": len(results),
            "results": [r.to_dict() for r in results]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """
    Get all analyses with optional filtering.
    
    Query parameters:
    - query: text search
    - sentiment: positive/neutral/negative
    - date_from: ISO format date
    - date_to: ISO format date
    - aspect: aspect name
    - stars: star rating (1-5)
    - limit: max results (default 100)
    """
    try:
        query = request.args.get("query", "")
        sentiment = request.args.get("sentiment", None)
        date_from = request.args.get("date_from", None)
        date_to = request.args.get("date_to", None)
        aspect = request.args.get("aspect", None)
        stars = request.args.get("stars", None, type=int)
        limit = request.args.get("limit", 100, type=int)
        
        # Search
        results = search_analyses(
            query=query,
            sentiment=sentiment,
            date_from=date_from,
            date_to=date_to,
            aspect=aspect,
            stars=stars
        )
        
        # Apply limit
        results = results[:limit]
        
        return jsonify({
            "success": True,
            "count": len(results),
            "analyses": [r.to_dict() for r in results]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/insights", methods=["GET"])
def get_insights():
    """
    Get aggregated insights from all analyses.
    Includes sentiment distribution, top aspects, trends, etc.
    """
    try:
        # Check cache first (optional optimization)
        # cached = load_insights_cache()
        # if cached:
        #     return jsonify({"success": True, "insights": cached})
        
        insights = compute_insights()
        save_insights_cache(insights)
        
        return jsonify({
            "success": True,
            "insights": insights
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export/csv", methods=["GET"])
def export_analyses_csv():
    """
    Export analyses to CSV file.
    
    Query parameters:
    - query: text search (optional)
    - sentiment: sentiment filter (optional)
    """
    try:
        query = request.args.get("query", "")
        sentiment = request.args.get("sentiment", None)
        
        # Search
        analyses = search_analyses(query=query, sentiment=sentiment)
        
        # Generate CSV
        csv_buffer = io.StringIO()
        if analyses:
            headers = ["review", "sentiment", "confidence", "stars", "created_at"]
            # Add aspect headers
            all_aspects = set()
            for a in analyses:
                all_aspects.update(a.aspect_scores.keys())
            aspect_headers = sorted(list(all_aspects))
            headers.extend(aspect_headers)
            
            csv_buffer.write(",".join(headers) + "\n")
            
            for analysis in analyses:
                row = [
                    f'"{analysis.review_text}"',
                    analysis.sentiment,
                    str(analysis.confidence),
                    str(analysis.stars),
                    analysis.created_at
                ]
                for aspect in aspect_headers:
                    score = analysis.aspect_scores.get(aspect, 0.0)
                    row.append(str(score))
                csv_buffer.write(",".join(row) + "\n")
        
        # Return as downloadable file
        csv_buffer.seek(0)
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete/<analysis_id>", methods=["DELETE"])
def delete(analysis_id):
    """Delete a specific analysis (not fully implemented - needs file path mapping)."""
    try:
        # TODO: Implement proper file tracking
        return jsonify({"success": True, "message": "Delete not yet implemented"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000, host="127.0.0.1")
