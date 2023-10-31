from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from step_counter.models.random.predict import main as predict_main
from step_counter.models.random.train import main as train_main


def test_main(dummy_train_and_test_data):
    train_path, test_path = dummy_train_and_test_data
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train_main,
            [
                str(train_path),
                "models/random/model.joblib",
                "metrics.json",
                "--seed",
                "100",
                "--kfolds",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path("models/random/model.joblib").exists()
        assert Path("metrics.json").exists()

        result = runner.invoke(
            predict_main,
            [
                str(test_path),
                "models/random/model.joblib",
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
