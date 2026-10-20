"""Reference classifier using CountVectorizer and scikit-learn MultinomialNB."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class SklearnMultinomialNB:
    """Expose the same fit/predict interface as the NumPy implementation."""

    def __init__(self, alpha=1.0, vocabulary=None):
        # Text has already been cleaned, so split on spaces without changing it.
        # An optional shared vocabulary ensures both models use the same words.
        self.vectorizer = CountVectorizer(
            vocabulary=vocabulary,
            lowercase=False,
            preprocessor=None,
            tokenizer=str.split,
            token_pattern=None,
        )
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, texts, labels):
        """Convert training sentences to word counts, then train Naive Bayes."""
        word_counts = self.vectorizer.fit_transform(texts)
        self.model.fit(word_counts, labels)
        return self

    def predict(self, texts):
        """Use the fitted vocabulary to classify new cleaned sentences."""
        word_counts = self.vectorizer.transform(texts)
        return self.model.predict(word_counts)

    def predict_proba(self, texts):
        """Return a probability for every class in each cleaned sentence."""
        word_counts = self.vectorizer.transform(texts)
        return self.model.predict_proba(word_counts)
