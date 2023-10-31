import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from step_counter.models.logistic_regression.predict import main as predict_main
from step_counter.models.logistic_regression.train import main as train_main


def test_main(dummy_train_and_test_data):
    train_path, test_path = dummy_train_and_test_data
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train_main,
            [
                str(train_path),
                "models/logistic_regression/model.joblib",
                "metrics.json",
                "--kfolds",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path("models/logistic_regression/model.joblib").exists()
        assert Path("metrics.json").exists()

        with open("metrics.json") as f:
            metrics = json.load(f)
            assert "train" in metrics, "'train' metrics not found"
            for metric_name, metric_value in metrics["train"].items():
                assert metric_value == 1.0, f"metric {metric_name} is not 1.0"
            assert "validation" in metrics, "'validation' metrics not found"

            assert (
                metrics["train"].keys() == metrics["validation"].keys()
            ), "train and validation metrics don't have the same keys"

        result = runner.invoke(
            predict_main,
            [
                str(test_path),
                "models/logistic_regression/model.joblib",
                "data/processed/test_data",
            ],
        )
        assert result.exit_code == 0, result.output
        output_files = list(Path("data/processed/test_data").glob("*.csv"))
        assert len(output_files) == 1

        for file in output_files:
            df = pd.read_csv(file)
            assert list(df.columns) == [
                "timestamp",
                "x",
                "y",
                "z",
                "button_state",
                "score",
            ], "output file has incorrect columns"

            assert len(df) == 4, "output file has incorrect number of rows"
