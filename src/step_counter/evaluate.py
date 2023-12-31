"""
Script to evaluate the classifier
"""
import json
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from step_counter.datasets import load_data_as_dataframe


def count_steps(button_state: np.ndarray) -> int:
    """Count the number of steps in a button state array."""
    # convolve with [-1, 1] to find edges
    kernel = np.array([-1, 1])
    convolved_bs = np.convolve(button_state, kernel, mode="same")
    # count the number of positive edges
    return np.equal(convolved_bs, 1).sum()


@click.command()
@click.argument("targets_path", type=click.Path(exists=True, path_type=Path))
@click.argument("predictions_path", type=click.Path(exists=True, path_type=Path))
@click.argument("figures_dir", type=click.Path(path_type=Path))
@click.argument("reports_dir", type=click.Path(path_type=Path))
def main(
    targets_path: Path, predictions_path: Path, figures_dir: Path, reports_dir: Path
):
    """
    Evaluate classifier on data
    TARGETS_PATH: path to input labeled data
    PREDICTIONS_PATH: path to model predictions
    FIGURES_DIR: path to save figures
    REPORTS_DIR: path to save reports

    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    # load data
    targets_data = load_data_as_dataframe(targets_path)
    y = targets_data["button_state"].values

    # load predictions
    predictions_data = load_data_as_dataframe(predictions_path)
    y_pred = predictions_data["button_state"].values
    y_scores = predictions_data["score"].values

    print("Running evaluation...")
    labels = ["no-step", "step"]

    with open(figures_dir / "confusion_matrix.csv", "w") as f:
        print("actual,predicted", file=f)
        for actual, predicted in zip(y.astype(int), y_pred.astype(int)):
            print(f"{labels[actual]},{labels[predicted]}", file=f)

    precision, recall, thresholds = precision_recall_curve(y, y_scores)

    with open(figures_dir / "precision_recall_curve.csv", "w") as f:
        print("precision,recall,threshold", file=f)
        for p, r, t in zip(precision, recall, thresholds):
            print(f"{p},{r},{t}", file=f)

    fpr, tpr, thresholds = roc_curve(
        y,
        y_scores,
    )
    with open(figures_dir / "roc_curve.csv", "w") as f:
        print("fpr,tpr,threshold", file=f)
        for fp, tp, t in zip(fpr, tpr, thresholds):
            print(f"{fp},{tp},{t}", file=f)

    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")

    cls_report = classification_report(y, y_pred, target_names=labels)
    print(cls_report)
    cls_report_dict = classification_report(
        y, y_pred, target_names=labels, output_dict=True
    )

    metrics = {
        "test": {f"{k}_macro": v for k, v in cls_report_dict["macro avg"].items()},
    }
    metrics["test"]["roc_auc"] = roc_auc

    target_steps_per_file = {}
    for file in targets_path.glob("*.csv"):
        target_data = load_data_as_dataframe(file.parent, glob_pattern=file.name)
        button_clicks = target_data["button_state"].values
        steps = count_steps(button_clicks)
        target_steps_per_file[file.name] = steps

    predicted_steps_per_file = {}
    for file in predictions_path.glob("*.csv"):
        predictions_data = load_data_as_dataframe(file.parent, glob_pattern=file.name)
        y_pred = predictions_data["button_state"].values
        steps = count_steps(y_pred)
        predicted_steps_per_file[file.name] = steps

    predicted_steps = int(sum(predicted_steps_per_file.values()))
    actual_steps = int(sum(target_steps_per_file.values()))
    print(f"Predicted steps: {predicted_steps}, actual steps: {actual_steps}")
    metrics["test"]["predicted_step_count"] = predicted_steps
    metrics["test"]["target_step_count"] = actual_steps

    with open(reports_dir / "test.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
