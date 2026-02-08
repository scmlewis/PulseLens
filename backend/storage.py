"""
Local file storage for analysis results and batch processing.
Stores results as JSON files, organized by analysis type and timestamp.
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from .models import AnalysisResult, BatchResult


# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
ANALYSES_DIR = DATA_DIR / "analyses"
BATCHES_DIR = DATA_DIR / "batches"
SHARES_DIR = DATA_DIR / "shares"
INSIGHTS_CACHE_FILE = DATA_DIR / "insights_cache.json"


def ensure_dirs():
    """Ensure all necessary directories exist."""
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    SHARES_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis(analysis: AnalysisResult) -> str:
    """
    Save a single analysis result to a JSON file.
    
    Args:
        analysis: AnalysisResult object
    
    Returns:
        Path to saved file
    """
    ensure_dirs()
    
    # Generate filename with timestamp
    timestamp = datetime.now().isoformat().replace(":", "-").split(".")[0]
    filename = f"{timestamp}_{analysis.sentiment}.json"
    filepath = ANALYSES_DIR / filename
    
    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
    
    return str(filepath)


def load_analysis(filepath: str) -> Optional[AnalysisResult]:
    """
    Load a single analysis result from a JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        AnalysisResult object or None if file not found
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AnalysisResult.from_dict(data)
    except Exception as e:
        print(f"Error loading analysis {filepath}: {e}")
        return None


def load_all_analyses() -> List[AnalysisResult]:
    """
    Load all analysis results from the analyses folder.
    
    Returns:
        List of AnalysisResult objects, sorted by date descending (newest first)
    """
    ensure_dirs()
    analyses = []
    
    try:
        for filepath in sorted(ANALYSES_DIR.glob("*.json"), reverse=True):
            result = load_analysis(str(filepath))
            if result:
                analyses.append(result)
    except Exception as e:
        print(f"Error loading analyses: {e}")
    
    return analyses


def save_batch(batch_id: str, batch_name: str, aspects: List[str], results: List[AnalysisResult]) -> str:
    """
    Save a batch processing result.
    
    Args:
        batch_id: Unique batch identifier
        batch_name: Human-readable batch name (e.g., filename)
        aspects: List of aspects used
        results: List of AnalysisResult objects
    
    Returns:
        Path to batch directory
    """
    ensure_dirs()
    
    batch_dir = BATCHES_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "num_reviews": len(results),
        "aspects_used": aspects,
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "completed_at": datetime.now().isoformat()
    }
    
    with open(batch_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Save results
    results_data = [r.to_dict() for r in results]
    with open(batch_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return str(batch_dir)


def load_batch(batch_id: str) -> tuple:
    """
    Load batch metadata and results.
    
    Args:
        batch_id: Unique batch identifier
    
    Returns:
        Tuple of (metadata_dict, list_of_AnalysisResult)
    """
    batch_dir = BATCHES_DIR / batch_id
    
    metadata = None
    results = []
    
    try:
        # Load metadata
        with open(batch_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Load results
        with open(batch_dir / "results.json", "r", encoding="utf-8") as f:
            results_data = json.load(f)
        
        results = [AnalysisResult.from_dict(r) for r in results_data]
    except Exception as e:
        print(f"Error loading batch {batch_id}: {e}")
    
    return metadata, results


def search_analyses(query: str = "", sentiment: str = None, 
                   date_from: str = None, date_to: str = None,
                   aspect: str = None, stars: int = None) -> List[AnalysisResult]:
    """
    Search and filter analyses based on multiple criteria.
    
    Args:
        query: Full-text search in review text (case-insensitive)
        sentiment: Filter by sentiment ("positive", "neutral", "negative")
        date_from: Filter by date (ISO format, e.g., "2026-01-01")
        date_to: Filter by date (ISO format)
        aspect: Filter by aspect (only returns reviews where aspect score > 0)
        stars: Filter by star rating (1-5)
    
    Returns:
        List of matching AnalysisResult objects
    """
    analyses = load_all_analyses()
    results = analyses
    
    # Text search
    if query:
        query_lower = query.lower()
        results = [r for r in results if query_lower in r.review_text.lower()]
    
    # Sentiment filter
    if sentiment:
        results = [r for r in results if r.sentiment == sentiment]
    
    # Date range filter
    if date_from or date_to:
        def parse_date(date_str):
            try:
                return datetime.fromisoformat(date_str)
            except:
                return None
        
        date_from_obj = parse_date(date_from) if date_from else None
        date_to_obj = parse_date(date_to) if date_to else None
        
        for result in results[:]:
            try:
                result_date = datetime.fromisoformat(result.created_at)
                if date_from_obj and result_date < date_from_obj:
                    results.remove(result)
                elif date_to_obj and result_date > date_to_obj:
                    results.remove(result)
            except:
                pass
    
    # Aspect filter
    if aspect:
        results = [r for r in results if aspect in r.aspect_scores and r.aspect_scores[aspect] > 0]
    
    # Star rating filter
    if stars:
        results = [r for r in results if r.stars == stars]
    
    return results


def compute_insights() -> dict:
    """
    Compute aggregated insights from all analyses.
    
    Returns:
        Dictionary with aggregated metrics (sentiment distribution, top aspects, trends, etc.)
    """
    analyses = load_all_analyses()
    
    if not analyses:
        return {
            "total_analyses": 0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "avg_sentiment_confidence": 0,
            "top_aspects": [],
            "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "sentiment_trend": [],
            "last_updated": datetime.now().isoformat()
        }
    
    # Sentiment distribution
    sentiment_dist = {"positive": 0, "neutral": 0, "negative": 0}
    total_confidence = 0
    
    # Aspect scores
    aspect_scores_sum = {}
    aspect_count = {}
    
    # Rating distribution
    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Sentiment trend (by week)
    trends_by_week = {}
    
    for analysis in analyses:
        # Sentiment distribution
        sentiment_dist[analysis.sentiment] += 1
        total_confidence += analysis.confidence
        
        # Aspect tracking
        for aspect, score in analysis.aspect_scores.items():
            if aspect not in aspect_scores_sum:
                aspect_scores_sum[aspect] = 0
                aspect_count[aspect] = 0
            aspect_scores_sum[aspect] += score
            aspect_count[aspect] += 1
        
        # Rating distribution
        rating_dist[analysis.stars] += 1
        
        # Trend by week
        try:
            week = datetime.fromisoformat(analysis.created_at).isocalendar()
            week_key = f"{week[0]}-W{week[1]:02d}"
            if week_key not in trends_by_week:
                trends_by_week[week_key] = {"positive": 0, "neutral": 0, "negative": 0}
            trends_by_week[week_key][analysis.sentiment] += 1
        except:
            pass
    
    # Top aspects by average score
    top_aspects = []
    for aspect, total_score in aspect_scores_sum.items():
        avg_score = total_score / aspect_count[aspect] if aspect_count[aspect] > 0 else 0
        top_aspects.append({"aspect": aspect, "avg_score": avg_score})
    
    top_aspects = sorted(top_aspects, key=lambda x: x["avg_score"], reverse=True)[:5]
    
    # Trend data
    sentiment_trend = [
        {"week": week, **counts}
        for week, counts in sorted(trends_by_week.items())
    ]
    
    return {
        "total_analyses": len(analyses),
        "sentiment_distribution": sentiment_dist,
        "avg_sentiment_confidence": total_confidence / len(analyses) if analyses else 0,
        "top_aspects": top_aspects,
        "rating_distribution": rating_dist,
        "sentiment_trend": sentiment_trend,
        "last_updated": datetime.now().isoformat()
    }


def save_insights_cache(insights: dict):
    """Save insights cache to file."""
    ensure_dirs()
    with open(INSIGHTS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)


def load_insights_cache() -> dict:
    """Load insights cache from file."""
    try:
        with open(INSIGHTS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def get_recent_analyses(limit: int = 10) -> List[AnalysisResult]:
    """Get most recent analyses."""
    return load_all_analyses()[:limit]


def delete_analysis(filepath: str) -> bool:
    """Delete an analysis result file."""
    try:
        os.remove(filepath)
        return True
    except:
        return False


def export_csv(filepath: str, analyses: List[AnalysisResult]):
    """
    Export analyses to CSV file.
    
    Args:
        filepath: Output CSV file path
        analyses: List of AnalysisResult objects
    """
    try:
        import pandas as pd
        
        # Flatten aspect scores into separate columns
        rows = []
        for analysis in analyses:
            row = {
                "review": analysis.review_text,
                "sentiment": analysis.sentiment,
                "confidence": analysis.confidence,
                "stars": analysis.stars,
                "created_at": analysis.created_at
            }
            for aspect, score in analysis.aspect_scores.items():
                row[f"aspect_{aspect}"] = score
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return False
