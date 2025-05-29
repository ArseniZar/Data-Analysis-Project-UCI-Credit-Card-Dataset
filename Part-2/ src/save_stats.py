from pathlib import Path
import pandas as pd
from typing import Union

def save_stats_to_csv(stats_df: pd.DataFrame, filename: Union[str, Path]) -> bool:
    if not isinstance(stats_df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(stats_df)}")

    try:
        stats_df.to_csv(filename)
        return True
    except PermissionError as e:
        print(f"Access error: No permission to write to file '{filename}'")
    except FileNotFoundError as e:
        print(f"Path error: Directory does not exist: '{Path(filename).parent}'")
    except pd.errors.EmptyDataError as e:
        print(f"Data error: DataFrame is empty, nothing to save")
    except Exception as e:
        print(f"Unknown error while saving file '{filename}': {str(e)}")
    return False