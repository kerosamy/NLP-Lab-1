"""Calculate metrics with NumPy, check them against sklearn, and save results."""

import csv
import json
from pathlib import Path

import numpy as np

MODEL_NAMES = {
    "numpy": "NumPy Naive Bayes",
    "sklearn": "scikit-learn Naive Bayes",
    "numpy_logistic": "NumPy Logistic Regression",
    "sklearn_logistic": "scikit-learn LogisticRegression",
    "sklearn_sgd": "scikit-learn SGDClassifier (log loss)",
}


def confusion_matrix(y_true, y_pred, labels):
    """Count predictions: rows are true classes, columns are predicted classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("true and predicted labels must be equally sized 1D arrays")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be nonempty and unique")

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label not in label_to_index or predicted_label not in label_to_index:
            raise ValueError("a true or predicted label is missing from labels")
        row = label_to_index[true_label]
        column = label_to_index[predicted_label]
        matrix[row, column] += 1

    return matrix


def safe_divide(numerator, denominator):
    """Divide element by element, returning zero wherever the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metrics_from_confusion_matrix(matrix):
    """Calculate per-class precision, recall, F1, macro averages, and accuracy."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("confusion matrix must be nonempty and square")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("confusion matrix counts must be finite and nonnegative")
    if np.any(matrix != np.floor(matrix)):
        raise ValueError("confusion matrix counts must be integers")

    true_positives = np.diag(matrix).astype(float)
    actual_counts = matrix.sum(axis=1)       # Row totals: actual sentences per class.
    predicted_counts = matrix.sum(axis=0)    # Column totals: predictions per class.
    false_positives = predicted_counts - true_positives
    false_negatives = actual_counts - true_positives

    # Precision = TP / (TP + FP), recall = TP / (TP + FN).
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    total_predictions = matrix.sum()
    accuracy = 0.0
    if total_predictions > 0:
        accuracy = float(true_positives.sum() / total_predictions)

    # Macro averages give every class equal weight, including rare classes.
    return {
        "confusion_matrix": matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_counts,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "accuracy": accuracy,
    }


def evaluate(y_true, y_pred, labels):
    """Build the confusion matrix, then derive all metrics from its counts."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    return metrics_from_confusion_matrix(matrix)


def validate_metrics(y_true, y_pred, result, labels):
    """Check the NumPy calculations against the required sklearn functions.

    sklearn is used only as a reference here, not to calculate our own metrics.
    """
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
    from sklearn.metrics import f1_score, precision_score, recall_score

    expected_matrix = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    np.testing.assert_array_equal(result["confusion_matrix"], expected_matrix)

    reference_functions = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    for metric_name, metric_function in reference_functions.items():
        expected_per_class = metric_function(
            y_true, y_pred, labels=labels, average=None, zero_division=0,
        )
        expected_macro = metric_function(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )
        np.testing.assert_allclose(result[metric_name], expected_per_class, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result["macro_" + metric_name], expected_macro, rtol=1e-12, atol=1e-12)


def evaluate_models(y_true, predictions_by_model, label_names):
    """Evaluate any classifiers using the same numeric labels (0, 1, ...)."""
    labels = np.arange(len(label_names))
    results = {}

    for model_name, predictions in predictions_by_model.items():
        model_results = evaluate(y_true, predictions, labels)
        validate_metrics(y_true, predictions, model_results, labels)
        results[model_name] = model_results

    return results


def format_evaluation(name, result, label_names):
    """Create one readable table for both the terminal and summary.txt."""
    lines = [
        f"\n{name}",
        f"{'Class':12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}",
    ]
    for index, label in enumerate(label_names):
        lines.append(
            f"{label:12} {result['precision'][index]:10.4f} "
            f"{result['recall'][index]:10.4f} {result['f1'][index]:10.4f} "
            f"{result['support'][index]:9d}"
        )

    lines.extend([
        f"{'Macro avg':12} {result['macro_precision']:10.4f} "
        f"{result['macro_recall']:10.4f} {result['macro_f1']:10.4f}",
        f"Accuracy: {result['accuracy']:.2%}",
        "Confusion matrix (rows=true, columns=predicted):",
        "Class order: " + ", ".join(label_names),
        str(result["confusion_matrix"]),
    ])
    return "\n".join(lines)


def print_evaluation(name, result, label_names):
    print(format_evaluation(name, result, label_names))


def save_confusion_plot(matrix, label_names, title, path):
    """Save a labeled PNG without opening a plot window."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 6.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, label="Number of sentences")
    axis.set(
        xticks=range(len(label_names)),
        yticks=range(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted emotion",
        ylabel="True emotion",
        title=title,
    )

    # Show each count; use white text on dark cells so it remains readable.
    threshold = matrix.max() / 2
    for row in range(len(label_names)):
        for column in range(len(label_names)):
            text_color = "white" if matrix[row, column] > threshold else "black"
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center", color=text_color,
            )
    figure.savefig(path, dpi=180)


def save_confusion_csv(path, matrix, label_names):
    """Write the same confusion matrix as a table with labeled rows and columns."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted", *label_names])
        for label, counts in zip(label_names, matrix):
            writer.writerow([label, *counts.tolist()])


def save_metrics_csv(path, results, label_names):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "class", "precision", "recall", "f1", "support"])

        for model_name, result in results.items():
            for index, label in enumerate(label_names):
                writer.writerow([
                    model_name, label,
                    result["precision"][index], result["recall"][index],
                    result["f1"][index], result["support"][index],
                ])
            writer.writerow([
                model_name, "macro_average",
                result["macro_precision"], result["macro_recall"],
                result["macro_f1"], int(result["support"].sum()),
            ])


def save_metrics_json(path, results, label_names, configuration):
    """Convert NumPy arrays to ordinary lists so JSON can store them."""
    serializable_results = {}
    for model_name, result in results.items():
        serializable_results[model_name] = {}
        for metric_name, value in result.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            serializable_results[model_name][metric_name] = value

    document = {
        "label_order": label_names,
        "configuration": configuration,
        "models": serializable_results,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def save_predictions_csv(path, texts, y_true, predictions, label_names):
    """Store each test sentence alongside its actual and predicted emotions."""
    model_names = list(predictions)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        prediction_columns = [f"{name}_prediction" for name in model_names]
        writer.writerow(["cleaned_text", "true_emotion", *prediction_columns])

        for index, text in enumerate(texts):
            true_emotion = label_names[int(y_true[index])]
            row = [text, true_emotion]
            for model_name in model_names:
                predicted_label = int(predictions[model_name][index])
                row.append(label_names[predicted_label])
            writer.writerow(row)


def save_summary(path, results, label_names, configuration):
    lines = [
        "Emotion classification results",
        f"Training: {configuration['train_size']:,}; test: {configuration['test_size']:,}",
    ]
    for model_name, result in results.items():
        display_name = MODEL_NAMES.get(model_name, model_name.replace("_", " "))
        lines.append(format_evaluation(display_name, result, label_names))

    lines.extend([
        f"\nPrediction agreement: {configuration['prediction_agreement']:.2%}",
        "All NumPy metrics were verified against scikit-learn.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(directory, results, predictions, texts, y_true, label_names, train_size):
    """Save matrices, metrics, and predictions for any number of classifiers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prediction_arrays = np.array(list(predictions.values()))
    all_models_agree = np.all(prediction_arrays == prediction_arrays[0], axis=0)
    configuration = {
        "train_size": train_size,
        "test_size": len(y_true),
        "prediction_agreement": float(np.mean(all_models_agree)),
        "sklearn_validation_passed": True,
    }

    for model_name, result in results.items():
        matrix = result["confusion_matrix"]
        display_name = MODEL_NAMES.get(model_name, model_name.replace("_", " "))
        save_confusion_csv(directory / f"{model_name}_confusion_matrix.csv", matrix, label_names)
        save_confusion_plot(
            matrix, label_names, display_name,
            directory / f"{model_name}_confusion_matrix.png",
        )

    save_metrics_csv(directory / "metrics.csv", results, label_names)
    save_metrics_json(directory / "metrics.json", results, label_names, configuration)
    save_predictions_csv(directory / "predictions.csv", texts, y_true, predictions, label_names)
    save_summary(directory / "summary.txt", results, label_names, configuration)
    return directory.resolve()
