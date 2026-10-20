"""scikit-learn reference classifiers for the shared bigram features."""

from sklearn.linear_model import LogisticRegression, SGDClassifier


class SklearnLogisticRegression:
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)


class SklearnSGDClassifier:
    def __init__(self, max_iter=1000, random_state=42):
        self.model = SGDClassifier(
            loss="log_loss", max_iter=max_iter, tol=1e-3,
            random_state=random_state,
        )

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)

