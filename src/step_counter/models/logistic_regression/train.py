import json
from pathlib import Path

import click
import numpy as np
from joblib import dump
from rich.console import Console
from rich.table import Table
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import estimator_html_repr

from step_counter.datasets import get_magnitude, load_data_as_dataframe


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

    X = data[["x", "y", "z"]]
    # add shift columns
    X["x-1"] = X["x"].shift(1).fillna(0).values
    X["y-1"] = X["y"].shift(1).fillna(0).values
    X["z-1"] = X["z"].shift(1).fillna(0).values

    X["x-2"] = X["x"].shift(2).fillna(0).values
    X["y-2"] = X["y"].shift(2).fillna(0).values
    X["z-2"] = X["z"].shift(2).fillna(0).values

    y = data["button_state"].values

    feature_engineering = ColumnTransformer(
        [
            ("magnitude", FunctionTransformer(get_magnitude), ["x", "y", "z"]),
            ("magnitude-1", FunctionTransformer(get_magnitude), ["x-1", "y-1", "z-1"]),
            ("magnitude-2", FunctionTransformer(get_magnitude), ["x-2", "y-2", "z-2"]),
        ]
    )

    model = make_pipeline(
        feature_engineering,
        preprocessing.StandardScaler(),
        LogisticRegression(
            random_state=seed,
            class_weight="balanced",
        ),
    )
    print("Training model with cross validation...")
    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(
        model, X, y, scoring=scoring, return_train_score=True, cv=kfolds, n_jobs=4
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
        "num_samples": len(X),
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

    # optimize prediction threshold using f1 score
    y_pred_proba = model.predict_proba(X)
    scores = []
    thresholds = np.arange(0, 1, 0.01)
    for threshold in thresholds:
        scores.append(f1_score(y, y_pred_proba[:, 1] > threshold, average="macro"))

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

    with open(model_save_path.with_suffix(".html"), "w") as f:
        f.write(estimator_html_repr(model))


if __name__ == "__main__":
    main()
