"""Multiclass logistic regression trained with NumPy minibatch SGD."""

import numpy as np


class NumpyLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=30, batch_size=128, seed=42):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = seed

    @staticmethod
    def _softmax(scores):
        scores = scores - scores.max(axis=1, keepdims=True) 
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def fit(self, features, labels):
        labels = np.asarray(labels)
        if features.shape[0] != len(labels):
            raise ValueError("features and labels must have the same number of rows")
        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        rows, feature_count = features.shape
        class_count = len(self.classes_)
        self.weights_ = np.zeros((feature_count, class_count))
        self.bias_ = np.zeros(class_count)
        self.loss_history_ = []

        generator = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            shuffled = generator.permutation(rows)
            epoch_loss = 0.0
            seen = 0
            for start in range(0, rows, self.batch_size):
                indices = shuffled[start:start + self.batch_size]
                batch = features[indices]
                targets = encoded[indices]
                errors = self._softmax(batch @ self.weights_ + self.bias_)
                epoch_loss -= np.log(
                    np.clip(errors[np.arange(len(indices)), targets], 1e-15, 1.0)
                ).sum()
                seen += len(indices)
                errors[np.arange(len(indices)), targets] -= 1.0
                errors /= len(indices)
                weight_gradient = np.asarray(batch.T @ errors)
                bias_gradient = errors.sum(axis=0)
                self.weights_ -= self.learning_rate * weight_gradient
                self.bias_ -= self.learning_rate * bias_gradient
            self.loss_history_.append(epoch_loss / seen)
        return self

    def predict_proba(self, features):
        if not hasattr(self, "weights_"):
            raise ValueError("fit the model before prediction")
        return self._softmax(features @ self.weights_ + self.bias_)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

