import time
from pathlib import Path
from typing import Tuple

import click
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from step_counter.datasets import load_data_as_dataframe


class KerasDNNPredictor:
    """Predict button state from accelerometer data using a trained Keras DNN model

    Args:
        model_path: Path to trained model

    """

    def __init__(self, model_path: Path):
        self.model = keras.models.load_model(model_path)
        self.model.summary()

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

        predictions = self.model.predict(dict(X), verbose=0)

        return float(predictions["decision"][0][0]), float(predictions["score"][0][0])


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
    model = KerasDNNPredictor(model_save_path)

    output_path.mkdir(parents=True, exist_ok=True)
    columns_to_save = ["timestamp", "x", "y", "z", "button_state", "score"]
    start = time.time()
    total = 0
    for file in data_path.glob("*.csv"):
        print(f"Predicting on file: {file}")
        df = load_data_as_dataframe(file.parent, glob_pattern=file.name)
        # iterate over rows of df and make predictions for each step
        for i, row in tqdm(df.iterrows(), total=len(df)):
            X = row[["x", "y", "z"]]
            y_pred, score = model.predict(X["x"], X["y"], X["z"])
            df.loc[i, "button_state"] = y_pred
            df.loc[i, "score"] = score
        total += len(df)
        df[columns_to_save].to_csv(output_path / file.name, index=False)
    elapsed = time.time() - start
    print(f"Prediction time: {elapsed:.2f}s, Speed {total/elapsed:.2f} steps/s")


if __name__ == "__main__":
    main()
