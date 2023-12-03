import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from step_counter.evaluate import count_steps
from step_counter.evaluate import main as evaluate_main


def test_count_steps():
    button_clicks = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    steps = count_steps(button_clicks)
    assert steps == 2


def test_main(dummy_train_and_test_data, tmp_path):
    _, test_path = dummy_train_and_test_data
    predictions_path = tmp_path / "predictions" / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    df = pd.DataFrame(
        columns=["timestamp", "button_state", "score"],
        data=[
            [0.0, 0, 0.1],
            [1.0, 1, 0.6],
            [2.0, 0, 0.3],
            [3.0, 1, 0.7],
        ],
    )
    df.to_csv(predictions_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        evaluate_main,
        [
            str(test_path),
            str(predictions_path.parent),
            str(figures),
            str(reports),
        ],
    )
    assert result.exit_code == 0, result.output
    output_files = list(reports.glob("*.json"))
    output_files = [str(file.relative_to(tmp_path)) for file in output_files]
    assert len(output_files) == 1
    assert output_files == ["reports/test.json"]

    data = json.loads((reports / "test.json").read_text())
    for metrics in [
        "precision_macro",
        "recall_macro",
        "f1-score_macro",
        "roc_auc",
    ]:
        assert metrics in data["test"]
    assert data["test"]["predicted_step_count"] == 1
    assert data["test"]["precision_macro"] == 1.0
    assert data["test"]["recall_macro"] == 1.0
    assert data["test"]["f1-score_macro"] == 1.0
    assert data["test"]["roc_auc"] == 1.0


# TODO test all zeros predictions
# TODO test all ones predictions
# TODO test 50% ones predictions
