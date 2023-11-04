import json
from pathlib import Path

import click
import numpy as np
from joblib import dump
from rich.console import Console
from rich.table import Table
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from step_counter.datasets import load_data_as_dataframe


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_save_path", type=click.Path(path_type=Path))
@click.argument("metrics_path", type=click.Path(path_type=Path))
@click.option("--seed", "-s", "seed", type=int, default=42, help="Random seed")
@click.option(
    "--kfolds",
    "-kf",
    "kfolds",
    type=int,
    default=5,
    help="Number of folds for CV training",
)
def main(
    data_path: Path,
    model_save_path: Path,
    metrics_path: Path,
    seed: int,
    kfolds: int,
):
    """
        Train and save logistic regression model

    DATA_PATH: path to input data
    MODEL_SAVE_PATH: path to save model
    METRICS_PATH: path to save training and validation metrics
    """
    # load data
    data = load_data_as_dataframe(data_path)

    X = data[["x", "y", "z", "magnitude"]].values
    y = data["button_state"].values
    model = make_pipeline(
        preprocessing.StandardScaler(),
        LogisticRegression(
            random_state=seed,
            class_weight="balanced",
        ),
    )
    print("Training model with cross validation...")
    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(
        model, X, y, scoring=scoring, return_train_score=True, cv=kfolds
    )
    train_scores = {metric: scores[f"train_{metric}"].mean() for metric in scoring}
    validation_scores = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
    table = Table(title="Evaluation metrics")
    table.add_column("Metric")
    table.add_column("Train")
    table.add_column("Validation")
    for metric in scoring:
        table.add_row(
            metric, f"{train_scores[metric]:.3f}", f"{validation_scores[metric]:.3f}"
        )
    console = Console()
    console.print(table)
    metrics = {
        "train": train_scores,
        "validation": validation_scores,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # re-train the model on the entire training set
    print("Training model on entire training set...")
    model.fit(X, y)
    print("Evaluating model on entire training set...")
    print(classification_report(y, model.predict(X)))
    y_pred_proba = model.predict_proba(X)
    # optimize prediction threshold using f1 score
    scores = []
    thresholds = np.arange(0, 1, 0.01)
    for threshold in thresholds:
        scores.append(f1_score(y, y_pred_proba[:, 1] > threshold))

    best_threshold = thresholds[np.argmax(scores)]
    print(f"Default threshold (0.5) f1-score: {scores[50]:.3f}")
    print(f"Best threshold: {best_threshold:.3f}, f1-score: {np.max(scores):.3f}")

    y_pred = y_pred_proba[:, 1] > best_threshold
    print("Final evaluation on entire training set using best threshold...")
    print(classification_report(y, y_pred))

    # save model
    print(f"Saving model to {model_save_path}")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_save_path)

    # save best threshold
    best_threshold_path = model_save_path.with_suffix(".json")
    print(f"Saving best threshold to {best_threshold_path}")
    with open(best_threshold_path, "w") as f:
        json.dump({"decision_threshold": best_threshold}, f, indent=4)


if __name__ == "__main__":
    main()
