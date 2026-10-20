# Part 2.2 Discussion

## Binary bigram features

Binary bigrams are simple, interpretable, and preserve local word order. They can
capture short phrases such as "not happy" that unigram features miss, and sparse
storage avoids allocating entries for absent features. Their limitations are
that they ignore word order beyond adjacent pairs, assign no importance to
repetition, and produce a large vocabulary with many rare features. They also
miss synonyms and longer-range context, so unseen bigrams at test time provide
no signal.

## Classifier comparison

No classifier is guaranteed to perform best on every split. Scikit-learn's
LogisticRegression often gives a strong, stable baseline because its solver
optimizes the objective carefully, while the NumPy minibatch implementation is
useful for understanding gradient updates and can be tuned. SGDClassifier can be
faster on very large sparse datasets and supports incremental training, but its
result depends more on learning-rate, regularization, and convergence settings.
Compare held-out macro F1 and accuracy using the same split and feature matrix
before concluding which performed better for this dataset.

SGDClassifier with `loss="log_loss"` is the closest algorithmic match to the
NumPy implementation: both use stochastic or minibatch gradient updates to
minimize logistic cross-entropy. LogisticRegression uses the same logistic
objective but typically relies on a batch or quasi-Newton solver instead.

