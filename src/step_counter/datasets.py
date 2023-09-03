import datetime
from pathlib import Path
from typing import Union

import pandas as pd


def load_data_as_dataframe(root_dir: Union[Path, str]) -> pd.DataFrame:
    """
    Load data from root_dir path into a single Pandas dataframe
    Args:
        root_dir: a path to the directory with data

    Returns:
        a single Pandas dataframe with data from all csv files found in root_dir

    """
    dfs = []
    for path in Path(root_dir).glob("*.csv"):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        )
        df = df.set_index("timestamp")
        dfs.append(df)

    df = pd.concat(dfs)
    return df
