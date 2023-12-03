import pandas as pd
import pytest

from step_counter.datasets import get_magnitude


def test__get_magnitude():
    df = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0.0, 0.1],
            [0.0, 0.2],
            [0.0, 0.3],
            [0.0, 0.4],
        ],
    )
    with pytest.raises(ValueError):
        get_magnitude(df)
