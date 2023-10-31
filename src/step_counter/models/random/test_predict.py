from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from step_counter.models.random.predict import main as predict_main
from step_counter.models.random.train import main as train_main


@pytest.fixture()
def dummy_train_and_test_data(tmp_path):
    train_path = tmp_path / "train_data"
    test_path = tmp_path / "test_data"
    train_path.mkdir()
    test_path.mkdir()
    dummy_df = pd.DataFrame(
        columns=["timestamp", "x", "y", "z", "button_state"],
        data=[
            [0.0, 0.0, 0.1, 0.0, 0],
            [1.0, 0.0, 0.2, 0.0, 1],
            [2.0, 0.0, 0.3, 0.0, 0],
            [3.0, 0.0, 0.4, 0.0, 1],
        ],
    )
    dummy_df.to_csv(train_path / "train.csv", index=False)
    dummy_df.to_csv(test_path / "test.csv", index=False)

    return train_path, test_path


def test_main(dummy_train_and_test_data):
    train_path, test_path = dummy_train_and_test_data
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train_main,
            [
                str(train_path),
                "models/random/model.joblib",
                "metrics",
                "--seed",
                "100",
                "--kfolds",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path("models/random/model.joblib").exists()
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
