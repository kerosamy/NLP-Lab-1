"""Multinomial Naive Bayes written from scratch using NumPy only."""

import numpy as np


class NumpyMultinomialNB:
    """Learn emotion probabilities from word counts in cleaned sentences.

    alpha controls additive smoothing. alpha=1 gives Laplace smoothing.
    Attributes ending in an underscore hold values learned during fit().
    """

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def fit(self, texts, labels):
        texts = list(texts)
        labels = np.array(labels)

        # 1. Get classes
        self.classes_ = np.unique(labels)

        # 2. Build vocabulary
        words = set()
        for text in texts:
            for word in text.split():
                words.add(word)

        self.vocabulary_ = {}
        for i, word in enumerate(sorted(words)):
            self.vocabulary_[word] = i

        num_classes = len(self.classes_)
        vocab_size = len(self.vocabulary_)

        # 3. Create counters
        self.class_count_ = np.zeros(num_classes)
        self.feature_count_ = np.zeros((num_classes, vocab_size))

        # 4. Count sentences and words for each class
        for text, label in zip(texts, labels):
            class_index = np.where(self.classes_ == label)[0][0]

            self.class_count_[class_index] += 1

            for word in text.split():
                word_index = self.vocabulary_[word]
                self.feature_count_[class_index][word_index] += 1

        # 5. Calculate P(class)
        class_prob = self.class_count_ / len(texts)
        self.class_log_prior_ = np.log(class_prob)

        # 6. Calculate P(word | class) with Laplace smoothing
        word_counts = self.feature_count_ + self.alpha

        total_words = word_counts.sum(axis=1, keepdims=True)

        word_prob = word_counts / total_words

        self.feature_log_prob_ = np.log(word_prob)

        return self

    def decision_function(self, texts):
        """Return a log posterior score for every sentence and class.

        score = log P(class) + sum of log P(word | class) for every word.
        Log probabilities avoid multiplying many very small numbers.
        """
        if not hasattr(self, "feature_log_prob_"):
            raise ValueError("fit the model before prediction")

        texts = list(texts)
        scores = np.tile(self.class_log_prior_, (len(texts), 1))

        for sentence_index, text in enumerate(texts):
            for word in text.split():
                # Ignore words that never appeared in training.
                if word in self.vocabulary_:
                    word_index = self.vocabulary_[word]
                    scores[sentence_index] += self.feature_log_prob_[:, word_index]

        # Empty or entirely unknown sentences keep their class prior scores.
        return scores

    def predict(self, texts):
        """Choose the class with the largest score for each sentence."""
        scores = self.decision_function(texts)
        best_class_indices = scores.argmax(axis=1)
        return self.classes_[best_class_indices]

    def predict_proba(self, texts):
        """Convert log scores to probabilities that sum to one per sentence."""
        scores = self.decision_function(texts)
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
