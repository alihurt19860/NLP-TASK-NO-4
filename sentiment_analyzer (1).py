"""
Sentiment Analyzer
-------------------
Reads a piece of text (movie review, tweet, comment, etc.) and predicts
whether it expresses Positive or Negative sentiment.

APPROACH
--------
A lexicon-based (rule-based) method is used instead of training a classifier
on a handful of examples:

  1. Text preprocessing (lowercasing, tokenizing on word boundaries).
  2. Every word is looked up in a built-in sentiment lexicon (a dictionary of
     ~250 common positive/negative words with a polarity weight).
  3. Negation words ("not", "never", "no", "n't"...) flip the polarity of the
     word that follows them ("not good" -> negative).
  4. Intensifiers ("very", "extremely"...) and diminishers ("slightly",
     "somewhat"...) scale the weight of the next word.
  5. All the (possibly flipped/scaled) scores are summed. Positive total ->
     "Positive", negative or zero total -> "Negative".

WHY THIS APPROACH
------------------
A machine-learning classifier (e.g. TF-IDF + Logistic Regression) is only as
good as its training data. A handful of hand-written example sentences is not
enough for it to generalize to new vocabulary it never saw in training - it
ends up memorizing specific phrases rather than learning sentiment. A curated
lexicon, by contrast, generalizes immediately to any sentence built from
common English sentiment words, which is what a real "does this app work on
sentences I type in" demo needs. (If you later have access to a large labeled
dataset - e.g. IMDB reviews - the same pipeline can be swapped for a trained
TF-IDF + Logistic Regression model; that code pattern is noted at the bottom
of this file.)

Run it:
    python3 sentiment_analyzer.py
Then type any sentence when prompted (or "quit" to exit).
"""

import re


# ---------------------------------------------------------------------------
# 1. SENTIMENT LEXICON
# ---------------------------------------------------------------------------
# word -> polarity weight. Larger magnitude = stronger sentiment.

POSITIVE_WORDS = {
    "good": 1.5, "great": 2.0, "excellent": 2.5, "amazing": 2.5, "awesome": 2.5,
    "fantastic": 2.5, "wonderful": 2.3, "brilliant": 2.2, "superb": 2.3,
    "outstanding": 2.3, "impressive": 1.8, "perfect": 2.5, "love": 2.3,
    "loved": 2.3, "loving": 2.0, "like": 1.2, "liked": 1.2, "enjoy": 1.8,
    "enjoyed": 1.8, "enjoyable": 1.7, "happy": 1.8, "glad": 1.4, "delighted": 2.2,
    "delightful": 2.0, "pleased": 1.5, "satisfied": 1.5, "satisfying": 1.5,
    "beautiful": 1.9, "lovely": 1.8, "nice": 1.2, "fun": 1.5, "funny": 1.2,
    "cool": 1.1, "smooth": 1.2, "fast": 1.0, "reliable": 1.3, "recommend": 1.6,
    "recommended": 1.6, "best": 2.2, "better": 1.0, "top": 1.2, "thrilled": 2.2,
    "exciting": 1.6, "excited": 1.6, "fascinating": 1.7, "engaging": 1.4,
    "helpful": 1.4, "kind": 1.2, "friendly": 1.3, "polite": 1.1, "professional": 1.1,
    "smart": 1.1, "clean": 1.1, "comfortable": 1.2, "affordable": 1.0, "worth": 1.0,
    "worthy": 1.0, "solid": 1.2, "gorgeous": 1.9, "stunning": 2.0, "breathtaking": 2.1,
    "unforgettable": 1.9, "memorable": 1.5, "charming": 1.6, "refreshing": 1.4,
    "smile": 1.3, "smiling": 1.3, "laugh": 1.2, "laughed": 1.2, "win": 1.4,
    "won": 1.4, "success": 1.5, "successful": 1.5, "resolved": 1.3, "fixed": 1.1,
    "quick": 1.0, "easy": 1.1, "intuitive": 1.2, "flawless": 2.2, "genius": 2.0,
    "masterpiece": 2.5, "brilliant": 2.2, "cheerful": 1.4, "grateful": 1.4,
    "thankful": 1.3, "proud": 1.4, "relaxing": 1.3, "peaceful": 1.2,
}

NEGATIVE_WORDS = {
    "bad": -1.5, "terrible": -2.5, "horrible": -2.5, "awful": -2.4, "worst": -2.6,
    "poor": -1.5, "disappointing": -2.0, "disappointed": -2.0, "disappointment": -1.9,
    "hate": -2.3, "hated": -2.3, "hating": -2.0, "dislike": -1.4, "disliked": -1.4,
    "annoying": -1.7, "annoyed": -1.6, "frustrating": -1.8, "frustrated": -1.7,
    "angry": -1.8, "furious": -2.3, "upset": -1.6, "sad": -1.5, "unhappy": -1.6,
    "boring": -1.6, "bored": -1.4, "dull": -1.3, "useless": -1.9, "broken": -1.7,
    "waste": -1.8, "wasted": -1.8, "regret": -1.7, "rude": -1.6, "unhelpful": -1.6,
    "slow": -1.1, "unreliable": -1.7, "overpriced": -1.4, "expensive": -0.8,
    "confusing": -1.4, "buggy": -1.6, "crash": -1.5, "crashes": -1.5,
    "crashed": -1.5, "crashing": -1.5, "ruined": -2.0, "ruin": -1.9, "dirty": -1.3,
    "smelled": -1.0, "smells": -1.0, "damaged": -1.4, "late": -0.8, "delay": -1.0,
    "delayed": -1.0, "delays": -1.0, "problem": -1.1, "problems": -1.1, "issue": -0.9,
    "issues": -0.9, "complaint": -1.2, "complain": -1.1, "ignored": -1.4,
    "ignoring": -1.3, "mediocre": -1.3, "pathetic": -2.1, "sloppy": -1.5,
    "cheap": -0.7, "cheaply": -1.2, "fake": -1.4, "scam": -2.3, "fail": -1.6,
    "failed": -1.6, "failure": -1.7, "unpleasant": -1.5, "uncomfortable": -1.3,
    "painful": -1.5, "painfully": -1.5, "creepy": -1.3, "disgusting": -2.2,
    "gross": -1.6, "nightmare": -2.0, "disaster": -2.1, "avoid": -1.3,
    "unacceptable": -1.9, "incompetent": -1.8, "overrated": -1.2, "flaw": -1.1,
    "flaws": -1.1, "flawed": -1.4, "cry": -1.1, "cried": -1.1, "crying": -1.1,
    "afraid": -1.2, "worried": -1.1, "worry": -1.0, "worrying": -1.1, "scared": -1.3,
}

NEGATIONS = {"not", "no", "never", "n't", "cannot", "cant", "can't", "won't",
             "wont", "isn't", "isnt", "doesn't", "doesnt", "didn't", "didnt",
             "wasn't", "wasnt", "aren't", "arent", "hardly", "barely", "without"}

INTENSIFIERS = {
    "very": 1.5, "extremely": 1.8, "really": 1.3, "so": 1.3, "totally": 1.5,
    "absolutely": 1.7, "completely": 1.6, "incredibly": 1.7, "super": 1.5,
    "truly": 1.4, "utterly": 1.6,
}

DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "kind": 0.7, "sort": 0.7, "a bit": 0.7,
    "barely": 0.5, "hardly": 0.5, "little": 0.7,
}


def tokenize(text: str):
    """Lowercase and split into words, keeping contractions like "n't" attached
    (e.g. "didn't" stays one token) so the negation list above matches."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def score_text(text: str):
    """Return (total_score, list_of_scored_tokens_for_debugging)."""
    tokens = tokenize(text)
    total = 0.0
    debug = []

    negate = False
    scale = 1.0

    for word in tokens:
        if word in NEGATIONS:
            negate = True
            continue
        if word in INTENSIFIERS:
            scale = INTENSIFIERS[word]
            continue
        if word in DIMINISHERS:
            scale = DIMINISHERS[word]
            continue

        weight = POSITIVE_WORDS.get(word) or NEGATIVE_WORDS.get(word)
        if weight is not None:
            adjusted = weight * scale
            if negate:
                adjusted = -adjusted
            total += adjusted
            debug.append((word, round(adjusted, 2)))
            negate = False
            scale = 1.0
        # Non-sentiment words (the, a, movie, etc.) don't reset negate/scale
        # immediately, so "not a good film" still negates "good".

    return total, debug


def predict_sentiment(text: str):
    """Return (sentiment_label, confidence_0_to_1, contributing_words)."""
    total, debug = score_text(text)
    sentiment = "Positive" if total > 0 else "Negative"

    # Turn the raw score into a 0.5-1.0 "confidence" style number using a
    # squashing function, purely for a friendlier display.
    confidence = 0.5 + min(abs(total) / 8.0, 0.5)

    return sentiment, confidence, debug


def main():
    print("Sentiment Analyzer (lexicon-based)")
    print("Type a sentence to analyze it. Type 'quit' to exit.\n")

    demo_texts = [
        "I absolutely loved this movie, the acting was fantastic!",
        "This was the worst experience of my life, totally disappointing.",
        "The food wasn't very good and the service was slow.",
        "Not a bad effort, but nothing special either.",
    ]
    print("Quick demo:")
    for t in demo_texts:
        sentiment, confidence, _ = predict_sentiment(t)
        print(f'  "{t}"\n    -> {sentiment} (confidence: {confidence:.0%})')
    print()

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        sentiment, confidence, debug = predict_sentiment(user_input)
        print(f"  -> Sentiment: {sentiment}  (confidence: {confidence:.0%})")
        if debug:
            print(f"     contributing words: {debug}")
        else:
            print("     (no strong sentiment words found - defaulted to Negative)")
        print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# OPTIONAL: swapping in a trainable ML model (if you have a large dataset)
# ---------------------------------------------------------------------------
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
#
# vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# X = vectorizer.fit_transform(list_of_texts)          # e.g. IMDB reviews
# model = LogisticRegression(max_iter=1000).fit(X, list_of_labels)
# model.predict(vectorizer.transform(["new review text"]))
