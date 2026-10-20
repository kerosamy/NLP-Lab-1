from pathlib import Path

from Evaluation.evaluation import MODEL_NAMES, evaluate_models, print_evaluation, save_results
from NaiveBayes.numpy_model import NumpyMultinomialNB
from NaiveBayes.sklearn_model import SklearnMultinomialNB
from splitting import LABEL_NAMES, load_and_split
from LogisticRegression.features import BigramVectorizer
from LogisticRegression.numpy_model import NumpyLogisticRegression
from LogisticRegression.sklearn_models import (
    SklearnLogisticRegression,
    SklearnSGDClassifier,
)


def main():
    # 1. Random split: 90% training, 10% testing, seed 42.
    train_texts, test_texts, train_labels, test_labels = load_and_split(test_size=0.1, seed=42)
    print(f"Training: {len(train_texts):,}; test: {len(test_texts):,}")

    # 2. Train the NumPy and scikit-learn models.
    numpy_model = NumpyMultinomialNB(alpha=1.0)
    numpy_model.fit(train_texts, train_labels)

    sklearn_model = SklearnMultinomialNB(alpha=1.0, vocabulary=numpy_model.vocabulary_)
    sklearn_model.fit(train_texts, train_labels)

    # 3. Build binary bigram features from training data only, then compare
    # three logistic regression implementations on identical sparse matrices.
    vectorizer = BigramVectorizer()
    train_bigrams = vectorizer.fit_transform(train_texts)
    test_bigrams = vectorizer.transform(test_texts)

    numpy_logistic = NumpyLogisticRegression()
    numpy_logistic.fit(train_bigrams, train_labels)
    sklearn_logistic = SklearnLogisticRegression()
    sklearn_logistic.fit(train_bigrams, train_labels)
    sklearn_sgd = SklearnSGDClassifier()
    sklearn_sgd.fit(train_bigrams, train_labels)

    # 4. Evaluate all models and save the results.
    predictions = {
        "numpy": numpy_model.predict(test_texts),
        "sklearn": sklearn_model.predict(test_texts),
        "numpy_logistic": numpy_logistic.predict(test_bigrams),
        "sklearn_logistic": sklearn_logistic.predict(test_bigrams),
        "sklearn_sgd": sklearn_sgd.predict(test_bigrams),
    }
    results = evaluate_models(test_labels, predictions, LABEL_NAMES)
    for model_name, metrics in results.items():
        print_evaluation(MODEL_NAMES[model_name], metrics, LABEL_NAMES)

    results_folder = Path(__file__).resolve().parent / "results"
    save_results(
        results_folder, results, predictions, test_texts, test_labels,
        LABEL_NAMES, train_size=len(train_texts),
    )
    print(f"\nResults saved to: {results_folder}")


if __name__ == "__main__":
    main()
