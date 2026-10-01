"""Metrics shared between validation and the final, separate test evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


def score_model(pipeline, X, y, detailed=False):
    probabilities = pipeline.predict_proba(X)
    predictions = pipeline.classes_[np.argmax(probabilities, axis=1)]
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=list(range(10)))),
    }
    if detailed:
        metrics["confusion_matrix"] = confusion_matrix(y, predictions, labels=list(range(10))).tolist()
        metrics["classification_report"] = classification_report(y, predictions, labels=list(range(10)),
                                                                 output_dict=True, zero_division=0)
    return metrics, predictions, probabilities
