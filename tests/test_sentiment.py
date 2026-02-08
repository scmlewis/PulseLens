def sentiment_to_stars_local(sentiment, score):
    if sentiment == "positive":
        if score >= 0.9:
            return 5
        elif score >= 0.75:
            return 4
        elif score >= 0.6:
            return 3
        else:
            return 3
    elif sentiment == "neutral":
        return 3
    else:
        if score >= 0.85:
            return 1
        elif score >= 0.6:
            return 2
        else:
            return 2


def test_positive_high():
    assert sentiment_to_stars_local('positive', 0.95) == 5


def test_positive_mid():
    assert sentiment_to_stars_local('positive', 0.8) == 4


def test_neutral():
    assert sentiment_to_stars_local('neutral', 0.5) == 3


def test_negative_high_conf():
    assert sentiment_to_stars_local('negative', 0.9) == 1


def test_negative_mid_conf():
    assert sentiment_to_stars_local('negative', 0.65) == 2
