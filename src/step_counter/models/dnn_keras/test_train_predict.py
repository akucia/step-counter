import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from step_counter.models.dnn_keras.predict import main as predict_main
from step_counter.models.dnn_keras.train import main as train_main


def test_main(dummy_train_and_test_data):
    train_path, test_path = dummy_train_and_test_data
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train_main,
            [
                str(train_path),
                "models/dnn/",
                "metrics.json",
                "--kfolds",
                "2",
                "-b",
                "2",
                "-e",
                "10",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert Path("models/dnn/variables").exists()
        assert Path("metrics.json").exists()

        with open("metrics.json") as f:
            metrics = json.load(f)
            assert "train" in metrics, "'train' metrics not found"
            for metric_name, metric_value in metrics["train"].items():
                assert metric_value >= 0.5, f"metric {metric_name} is not >= 0.5"
            assert "validation" in metrics, "'validation' metrics not found"

            train_metrics = metrics["train"].keys()
            validation_metrics = metrics["validation"].keys()
            validation_metrics = {m for m in validation_metrics if "threshold" not in m}
            assert (
                train_metrics == validation_metrics
            ), "train and validation metrics don't have the same keys"

        result = runner.invoke(
            predict_main,
            [
                str(test_path),
                "models/dnn/",
                "data/processed/test_data",
            ],
            catch_exceptions=False,
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
