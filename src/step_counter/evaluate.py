"""
Script to evaluate the classifier
"""
from pathlib import Path

import click

from step_counter.datasets import load_data_as_dataframe


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
def main(data_path: Path):
    """
    Evaluate classifier on data
    DATA_PATH: path to input data
    """
    # load data
    data = load_data_as_dataframe(data_path)
    X = data[["x", "y", "z"]].values
    y = data["button_state"].values
    _ = X
    _ = y
    print("Running evaluation...")
    # TODO evaluate predictions with binary classification metrics
    # TODO plot confusion matrix
    # TODO save metrics and confusion matrix plot
    # TODO generate and save classification report
    # TODO generate and save precision-recall curve for dvc
    # TODO generate and save ROC curve for dvc


if __name__ == "__main__":
    main()
