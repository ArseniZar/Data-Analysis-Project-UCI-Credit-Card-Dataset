from typing import Dict, Any, List
import pandas as pd
import numpy as np

MEAN = 'mean'
MEDIAN = 'median'
MIN = 'min'
MAX = 'max'
STD = 'std'
FIFTH_PERCENTILE = '5th_percentile'
NINETY_FIFTH_PERCENTILE = '95th_percentile'
MISSING_VALUES = 'missing_values'

UNIQUE_CLASSES = 'unique_classes'
CLASS_PROPORTIONS = 'class_proportions'




def calculate_statistics(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Received None instead of DataFrame")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas.DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not isinstance(numeric_cols, list) or not isinstance(categorical_cols, list):
        raise ValueError("numeric_cols and categorical_cols must be lists")

    missing_numeric = [col for col in numeric_cols if col not in df.columns]
    missing_categorical = [col for col in categorical_cols if col not in df.columns]
    if missing_numeric:
        raise ValueError(f"Numeric columns not found in DataFrame: {missing_numeric}")
    if missing_categorical:
        raise ValueError(
            f"Categorical columns not found in DataFrame: {missing_categorical}"
        )

    overlapping_cols = set(numeric_cols).intersection(set(categorical_cols))
    if overlapping_cols:
        raise ValueError(
            f"Columns cannot be both numeric and categorical: {overlapping_cols}"
        )

    stats = {}

    for column in numeric_cols:
        try:
            stats[column] = {
                MEAN: df[column].mean(),
                MEDIAN: df[column].median(),
                MIN: df[column].min(),
                MAX: df[column].max(),
                STD: df[column].std(),
                FIFTH_PERCENTILE: df[column].quantile(0.05),
                NINETY_FIFTH_PERCENTILE: df[column].quantile(0.95),
                MISSING_VALUES: df[column].isna().sum(),
            }
        except Exception as e:
            raise ValueError(f"Error processing numeric column '{column}': {str(e)}")

    for column in categorical_cols:
        try:
            stats[column] = {
                UNIQUE_CLASSES: df[column].nunique(),
                MISSING_VALUES: df[column].isna().sum(),
                CLASS_PROPORTIONS: df[column]
                .value_counts(normalize=True)
                .to_dict(),
            }
        except Exception as e:
            raise ValueError(
                f"Error processing categorical column '{column}': {str(e)}"
            )

    stats_df = pd.DataFrame(stats).T
    return stats_df
