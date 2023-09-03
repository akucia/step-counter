from pathlib import Path

import click
import pandas as pd
from joblib import load


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_save_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def main(
    data_path: Path,
    model_save_path: Path,
    output_path: Path,
):
    """
    Predicts the button state for each row in the data_path csv file using random predictions.

    DATA_PATH: Path to the csv file containing the data to predict on.
    MODEL_SAVE_PATH: Path to the model to load.
    OUTPUT_PATH: Path to save the predictions to.

    """
    print(f"Loading model from {model_save_path}...")
    model = load(model_save_path)
    print(model)

    output_path.mkdir(parents=True, exist_ok=True)

    for file in data_path.glob("*.csv"):
        print(f"Predicting on file: {file}")
        df = pd.read_csv(file)
        X = df[["x", "y", "z"]].values
        y_pred = model.predict(X)
        df["button_state"] = y_pred
        df["score"] = model.predict_proba(X).max(axis=1)
        df.to_csv(output_path / file.name, index=False)


if __name__ == "__main__":
    main()
