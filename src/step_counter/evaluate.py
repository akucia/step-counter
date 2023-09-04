"""
Script to evaluate the classifier
"""
import json
from pathlib import Path

import click
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from step_counter.datasets import load_data_as_dataframe


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

    # TODO evaluate predictions with binary classification metrics
    cls_report = classification_report(y, y_pred, target_names=labels)
    print(cls_report)
    cls_report_dict = classification_report(
        y, y_pred, target_names=labels, output_dict=True
    )
    metrics = {
        "test": {f"{k}_macro": v for k, v in cls_report_dict["macro avg"].items()},
    }
    with open(reports_dir / "test.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # TODO plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    # save plot
    plt.savefig(figures_dir / "confusion_matrix.png", bbox_inches="tight")

    cm_norm = confusion_matrix(y, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot()
    # save plot
    plt.savefig(figures_dir / "confusion_matrix_normalized.png", bbox_inches="tight")

    roc_display = RocCurveDisplay.from_predictions(y, y_scores)
    roc_display.plot()
    plt.savefig(figures_dir / "roc_curve.png", bbox_inches="tight")

    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_disp.plot()
    plt.savefig(figures_dir / "precision_recall_curve.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
