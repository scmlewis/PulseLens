"""
Data models and classification logic for PulseLens backend.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import json


@dataclass
class AnalysisResult:
    """Single review analysis result."""
    review_text: str
    sentiment: str
    confidence: float
    stars: int
    aspects: List[str]
    aspect_scores: Dict[str, float]
    created_at: str
    analysis_id: Optional[str] = None
    title: Optional[str] = None
    is_favorite: bool = False
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class BatchResult:
    """Batch processing result metadata."""
    batch_id: str
    batch_name: str
    num_reviews: int
    aspects_used: List[str]
    created_at: str
    completed_at: Optional[str] = None
    status: str = "processing"  # processing, completed, failed
    
    def to_dict(self):
        return asdict(self)


def sentiment_to_stars(sentiment: str, confidence: float) -> int:
    """
    Map sentiment label and confidence score to a 1-5 star rating.
    
    Args:
        sentiment: One of "positive", "neutral", "negative"
        confidence: Confidence score from 0.0 to 1.0
    
    Returns:
        Star rating (1-5)
    """
    if sentiment == "positive":
        if confidence >= 0.9:
            return 5
        elif confidence >= 0.75:
            return 4
        elif confidence >= 0.6:
            return 3
        else:
            return 3
    elif sentiment == "neutral":
        return 3
    else:  # negative
        if confidence >= 0.85:
            return 1
        elif confidence >= 0.6:
            return 2
        else:
            return 2


def classify_sentiment(review_text: str, classifier) -> tuple:
    """
    Classify review sentiment using zero-shot classification.
    
    Args:
        review_text: Customer review text
        classifier: HuggingFace pipeline object
    
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    sentiment_labels = ["positive", "neutral", "negative"]
    try:
        result = classifier(review_text, candidate_labels=sentiment_labels, multi_label=False)
        # HF returns labels and scores in order of relevance
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return top_label, float(top_score)
    except Exception as e:
        print(f"Error classifying sentiment: {e}")
        return "neutral", 0.5


def classify_aspects(review_text: str, aspects: List[str], classifier) -> Dict[str, float]:
    """
    Classify aspect relevance scores for a review.
    
    Args:
        review_text: Customer review text
        aspects: List of aspect labels to classify against
        classifier: HuggingFace pipeline object
    
    Returns:
        Dictionary mapping aspect names to relevance scores (0.0-1.0)
    """
    if not aspects:
        return {}
    
    scores = {}
    try:
        result = classifier(review_text, candidate_labels=aspects, multi_label=True)
        # Multi-label returns all labels with scores
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] = float(score)
        # Ensure all aspects are in the dict (with 0 score if not relevant)
        for aspect in aspects:
            if aspect not in scores:
                scores[aspect] = 0.0
    except Exception as e:
        print(f"Error classifying aspects: {e}")
        for aspect in aspects:
            scores[aspect] = 0.0
    
    return scores


def classify_batch(reviews: List[str], aspects: List[str], classifier, chunk_size: int = 32):
    """
    Classify a batch of reviews.
    
    Args:
        reviews: List of review texts
        aspects: List of aspects to classify
        classifier: HuggingFace pipeline object
        chunk_size: Number of reviews to process per batch
    
    Returns:
        List of AnalysisResult objects
    """
    results = []
    timestamp = datetime.now().isoformat()
    
    for i, review in enumerate(reviews):
        # Classify sentiment
        sentiment, confidence = classify_sentiment(review, classifier)
        stars = sentiment_to_stars(sentiment, confidence)
        
        # Classify aspects
        aspect_scores = classify_aspects(review, aspects, classifier)
        
        # Create result
        result = AnalysisResult(
            review_text=review,
            sentiment=sentiment,
            confidence=confidence,
            stars=stars,
            aspects=aspects,
            aspect_scores=aspect_scores,
            created_at=timestamp,
            analysis_id=f"batch_{i}",
            title=f"Review_{timestamp.replace(':', '-').split('.')[0]}_{sentiment}"
        )
        results.append(result)
    
    return results


# Grouped aspects by industry
GROUPED_ASPECTS = {
    "🍽️ Restaurant": ["food", "service", "ambience", "price", "delivery", "staff", "product quality"],
    "💻 Electronics": ["battery", "display", "camera", "performance", "durability", "shipping", "support"],
    "👗 Fashion": ["fit", "material", "style", "comfort", "design", "price"],
    "🛒 Supermarket": ["freshness", "variety", "checkout", "customer service", "packaging", "speed"],
    "📚 Books": ["plot", "characters", "writing", "pacing", "ending", "value"],
    "🏨 Hotel": ["cleanliness", "location", "amenities", "room", "wifi", "maintenance"]
}


SAMPLE_COMMENTS = [
    "I visited the restaurant last night and was impressed by the cozy ambience and friendly staff. The food was delicious, especially the pasta, but the wait time for our main course was a bit long. Overall, a pleasant experience and I would recommend it to friends.",
    "This smartphone has a stunning display and the battery lasts all day, even with heavy use. However, the camera struggles in low light and the device sometimes gets warm during gaming sessions. Customer support was helpful when I had questions about the warranty.",
    "The dress I ordered online arrived quickly and the material feels premium. The fit is true to size and the color matches the photos perfectly. I received several compliments at the event, but I wish the price was a bit lower.",
    "Shopping at this supermarket is always convenient. The produce section is well-stocked and the staff are courteous. However, the checkout lines can get long during weekends and some items are more expensive compared to other stores.",
    "This novel captivated me from the first page. The plot twists kept me guessing, and the characters were well-developed. The pacing slowed down in the middle, but the ending was satisfying. Highly recommended for fans of mystery and drama.",
    "Our stay at the hotel was comfortable. The room was clean and spacious, and the staff were attentive to our needs. The breakfast buffet had a good variety, but the Wi-Fi connection was unreliable at times. The location is perfect for sightseeing."
]
