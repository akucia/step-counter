import json
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from step_counter.datasets import load_data_as_dataframe


class LogisticRegressionPredictor:
    """Predict button state from accelerometer data using a trained LogisticRegression model

    Notes:
        This class stores previous accelerometer data to use as input for the next prediction.

    """

    def __init__(self, model_path: Path):
        self.model = load(model_path)
        if isinstance(self.model, Pipeline):
            for i, step in enumerate(self.model.steps):
                print(f"Step {i}: {step}")
        else:
            print(self.model)
        with open(model_path.with_suffix(".json")) as f:
            model_metadata = json.load(f)
            self.decision_threshold = model_metadata["decision_threshold"]
        self.input = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    def predict(self, x, y, z: float) -> Tuple[float, float]:
        """Predict button state from accelerometer data

        Args:
            x: accelerometer data from x axis
            y: accelerometer data from y axis
            z: accelerometer data from z axis

        Returns:
            Tuple with predicted button state and prediction score


        """
        self.input.append((x, y, z))
        self.input.pop(0)

        X = pd.DataFrame(
            np.array(self.input).reshape(1, 9),
            columns=["x-2", "y-2", "z-2", "x-1", "y-1", "z-1", "x", "y", "z"],
        )

        y_pred_proba = self.model.predict_proba(X)[0, 1]
        return (y_pred_proba > self.decision_threshold).astype(float), y_pred_proba


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_save_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def main(
    data_path: Path,
    model_save_path: Path,
    output_path: Path,
):
    """Predict button state from accelerometer data using a trained LogisticRegression model

    DATA_PATH: Path to directory containing CSV files with accelerometer data
    MODEL_SAVE_PATH: Path to trained model
    OUTPUT_PATH: Path to directory to save CSV files with predictions

    """
    print(f"Loading model from {model_save_path}...")
    model = LogisticRegressionPredictor(model_save_path)

    output_path.mkdir(parents=True, exist_ok=True)
    columns_to_save = ["timestamp", "x", "y", "z", "button_state", "score"]
    for file in data_path.glob("*.csv"):
        print(f"Predicting on file: {file}")
        df = load_data_as_dataframe(file.parent, glob_pattern=file.name)
        # iterate over rows of df and make predictions for each step
        for i, row in df.iterrows():
            X = row[["x", "y", "z"]]
            y_pred, score = model.predict(X["x"], X["y"], X["z"])
            df.loc[i, "button_state"] = y_pred
            df.loc[i, "score"] = score

        df[columns_to_save].to_csv(output_path / file.name, index=False)


if __name__ == "__main__":
    main()
