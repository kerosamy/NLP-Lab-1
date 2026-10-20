"""Binary bigram feature extraction for preprocessed sentences."""

import numpy as np
from scipy.sparse import csr_matrix


class BigramVectorizer:
    """Map sentences to sparse binary vectors using a training-set vocabulary."""

    def fit(self, texts):
        bigrams = set()
        for text in texts:
            tokens = text.split()
            bigrams.update(zip(tokens, tokens[1:]))
        self.vocabulary_ = {
            bigram: index for index, bigram in enumerate(sorted(bigrams))
        }
        return self

    def transform(self, texts):
        if not hasattr(self, "vocabulary_"):
            raise ValueError("fit the vectorizer before transform")
        row_indices, column_indices = [], []
        for row, text in enumerate(texts):
            tokens = text.split()
            columns = {
                self.vocabulary_[bigram]
                for bigram in zip(tokens, tokens[1:])
                if bigram in self.vocabulary_
            }
            row_indices.extend([row] * len(columns))
            column_indices.extend(columns)
        data = np.ones(len(row_indices), dtype=np.float64)
        return csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(texts), len(self.vocabulary_)), dtype=np.float64,
        )

    def fit_transform(self, texts):
        texts = list(texts)
        self.fit(texts)
        return self.transform(texts)

