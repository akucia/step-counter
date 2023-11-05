from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def load_data_as_dataframe(
    root_dir: Union[Path, str], glob_pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Load data from root_dir path into a single Pandas dataframe
    Args:
        root_dir: a path to the directory with data
        glob_pattern: a glob pattern to match files in root_dir

    Returns:
        a single Pandas dataframe with data from all csv files found in root_dir

    """
    dfs = []
    for path in Path(root_dir).glob(glob_pattern):
        df = pd.read_csv(path)
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def get_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    df["magnitude"] = np.linalg.norm(df[["x", "y", "z"]].values, axis=1)
    return df
