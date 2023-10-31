from pathlib import Path

import click
from joblib import load
from sklearn.pipeline import Pipeline

from step_counter.datasets import load_data_as_dataframe


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
    model = load(model_save_path)
    if isinstance(model, Pipeline):
        for i, step in enumerate(model.steps):
            print(f"Step {i}: {step}")
    else:
        print(model)

    output_path.mkdir(parents=True, exist_ok=True)
    columns_to_save = ["timestamp", "x", "y", "z", "button_state", "score"]
    for file in data_path.glob("*.csv"):
        print(f"Predicting on file: {file}")
        df = load_data_as_dataframe(file.parent, glob_pattern=file.name)
        X = df[["x", "y", "z", "magnitude"]].values
        y_pred = model.predict(X)
        df["button_state"] = y_pred
        df["score"] = model.predict_proba(X)[:, 1]

        df[columns_to_save].to_csv(output_path / file.name, index=False)


if __name__ == "__main__":
    main()
