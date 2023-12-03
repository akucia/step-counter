"""Trains and saves logistic regression model for step counter using keras"""

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import click
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.src.layers import Normalization
from keras.src.utils import plot_model
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from tqdm import tqdm

from step_counter.datasets import load_data_as_dataframe

features = [
    "x",
    "y",
    "z",
    "x-1",
    "y-1",
    "z-1",
    "x-2",
    "y-2",
    "z-2",
]


def dataframes_to_dataset(X: pd.DataFrame, y: pd.DataFrame) -> tf.data.Dataset:
    """Converts pandas dataframes to tensorflow dataset

    Args:
        X (pd.DataFrame): features
        y (pd.DataFrame): target
    """
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    ds = ds.shuffle(buffer_size=len(X))
    return ds


def encode_numerical_feature(
    feature: tf.keras.Input, name: str, dataset: tf.data.Dataset
) -> tf.keras.layers.Layer:
    """Normalizes numerical features with keras"""
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


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
    # set seed with keras
    keras.utils.set_random_seed(seed)

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

    y = data["button_state"].values.reshape(-1, 1)

    scores = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    splits = kf.split(X, y=y)

    thresholds = []

    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        futures = []
        for train, test in splits:
            X_train, X_test, y_train, y_test = (
                X.iloc[train],
                X.iloc[test],
                y[train],
                y[test],
            )
            futures.append(
                executor.submit(_train_model, X_test, X_train, y_test, y_train, seed)
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Training"):
            task_scores, task_threshold = future.result()
            thresholds.append(task_threshold)
            for metric, score in task_scores["train"].items():
                scores["train"][metric].append(score)
            for metric, score in task_scores["val"].items():
                scores["val"][metric].append(score)

    threshold_mean, threshold_std = float(np.mean(thresholds)), float(
        np.std(thresholds)
    )

    train_scores = {
        metric: np.mean(scores) for metric, scores in scores["train"].items()
    }
    validation_scores = {
        metric: np.mean(scores) for metric, scores in scores["val"].items()
    }
    validation_scores["threshold_mean"] = threshold_mean
    validation_scores["threshold_std"] = threshold_std

    table = Table(title="Average evaluation metrics")
    table.add_column("Metric")
    table.add_column("Train")
    table.add_column("Validation")

    for metric in sorted(train_scores.keys() | validation_scores.keys()):
        table.add_row(
            metric,
            f"{train_scores.get(metric, -1):.3f}",
            f"{validation_scores.get(metric, -1):.3f}",
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
    train_ds = dataframes_to_dataset(X, y)
    train_ds = train_ds.batch(512)
    model = _build_keras_model(train_ds, features)
    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y), y=y.flatten()
    )
    cw = dict(enumerate(cw))

    model.fit(
        train_ds,
        epochs=150,
        verbose=0,
        class_weight=cw,
    )
    print("Evaluating model on entire training set...")
    y_true, y_pred = _predict_on_dataset(model, train_ds, threshold=threshold_mean)

    print("Final evaluation on entire training set using best threshold...")
    print(classification_report(y_true, y_pred))

    # add layer with threshold to the model
    x = model(model.inputs)
    threshold_scores = layers.Lambda(lambda output: output > threshold_mean)(x)
    model = keras.Model(
        inputs=model.inputs, outputs={"score": x, "decision": threshold_scores}
    )
    model.summary()

    # save model
    print(f"Saving model to {model_save_path}")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path, save_format="tf")

    plot_model(
        model,
        to_file=model_save_path / "model.png",
        show_shapes=True,
        rankdir="LR",
    )


def _find_optimal_threshold(y_true, y_pred, scoring_fn: Callable = f1_score) -> float:
    print("Finding optimal threshold...")
    # optimize prediction threshold using f1 score
    scores = []
    thresholds = np.arange(0, 1, 0.01)
    for threshold in thresholds:
        scores.append(scoring_fn(y_true, y_pred > threshold, average="macro"))
    best_threshold, best_value = thresholds[np.argmax(scores)], np.max(scores)
    print(f"Default threshold (0.5) f1-score_macro: {scores[50]:.3f}")
    print(f"Best threshold: {best_threshold:.3f}, f1-score: {best_value:.3f}")
    return float(best_threshold)


def _train_model(
    X_test, X_train, y_test, y_train, seed
) -> Tuple[Dict[str, Dict[str, float]], float]:
    # TODO better typing for return value
    keras.utils.set_random_seed(seed)
    scores = {
        "train": dict(),
        "val": dict(),
    }
    train_ds = dataframes_to_dataset(X_train, y_train)
    val_ds = dataframes_to_dataset(X_test, y_test)
    train_ds = train_ds.batch(512)  # TODO change to script args
    val_ds = val_ds.batch(512)

    model = _build_keras_model(train_ds, features)
    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train.flatten()
    )
    cw = dict(enumerate(cw))
    model.fit(
        train_ds,
        epochs=150,
        validation_data=val_ds,
        verbose=0,
        class_weight=cw,
    )
    train_eval_threshold = 0.5
    y_true, y_pred = _predict_on_dataset(
        model, train_ds, threshold=train_eval_threshold
    )

    train_metrics = _calculate_metrics(y_true, y_pred)
    for metric, score in train_metrics.items():
        scores["train"][metric] = score

    y_true, y_pred = _predict_on_dataset(model, val_ds, threshold=None)

    best_threshold = _find_optimal_threshold(y_true, y_pred)

    val_metrics = _calculate_metrics(y_true, y_pred > best_threshold)
    for metric, score in val_metrics.items():
        scores["val"][metric] = score
    return scores, best_threshold


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def _predict_on_dataset(
    model: keras.Model, dataset: tf.data.Dataset, threshold: Optional[float] = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for batch_x, batch_y in dataset:
        y_true.append(batch_y.numpy())
        y_pred.append(model.predict(batch_x, verbose=0))
    y_pred = np.concatenate(y_pred)
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(float)

    y_true = np.concatenate(y_true)
    return y_true, y_pred


def _build_keras_model(
    train_ds: tf.data.Dataset,
    features: List[str],
    units: int = 100,
    num_layers: int = 1,
    dropout: float = 0.5,
    learning_rate=5e-3,
    compile: bool = True,
):
    # build model
    x_input = []
    encoded_features = {}

    for header in features:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        encoded_feature = encode_numerical_feature(numeric_col, header, train_ds)
        x_input.append(numeric_col)
        encoded_features[header] = encoded_feature

    # add magnitude features as sqrt (x**2 + y**2 + z**2)
    encoded_features["magnitude"] = layers.Lambda(
        lambda x: tf.sqrt(x["x"] ** 2 + x["y"] ** 2 + x["z"] ** 2)
    )(encoded_features)
    encoded_features["magnitude-1"] = layers.Lambda(
        lambda x: tf.sqrt(x["x-1"] ** 2 + x["y-1"] ** 2 + x["z-1"] ** 2)
    )(encoded_features)
    encoded_features["magnitude-2"] = layers.Lambda(
        lambda x: tf.sqrt(x["x-2"] ** 2 + x["y-2"] ** 2 + x["z-2"] ** 2)
    )(encoded_features)

    all_features = layers.concatenate(encoded_features.values())
    x = all_features
    for _ in range(num_layers):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(x_input, output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if compile:
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.BinaryAccuracy(),
            ],
        )
    return model


if __name__ == "__main__":
    main()
