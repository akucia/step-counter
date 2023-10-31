from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest


@pytest.fixture()
def dummy_train_and_test_data(tmp_path: Path) -> Tuple[Path, Path]:
    """Create dummy train and test data in .csv files

    Args:
        tmp_path (Path): pytest fixture for temporary directory

    Notes:
        csv files contain 4 rows of dummy data with the following columns:
        - timestamp: float
        - x: float
        - y: float
        - z: float
        - button_state: int
    """
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
