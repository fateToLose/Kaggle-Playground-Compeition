import pandas as pd
import numpy as np

from typing import Any

from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer


# -----  Global ----- #
class PreproceesError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


# -----  Functions ----- #
def prepare_data(data: pd.DataFrame, target_col: str = "y") -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target_col in data.columns:
        X = data.drop([target_col], axis=1)
        Y = data.loc[:, target_col].copy()
    else:
        raise PreproceesError(f"Target col ({target_col}) not found in dataframe.")

    if "id" in X.columns:
        X = X.drop(["id"], axis=1)

    numerical_cols = X.select_dtypes(include=[np.number]).columns.to_list()
    category_cols = X.select_dtypes(include=["object"]).columns.to_list()

    return X, Y, numerical_cols, category_cols


def feature_engineering(numerical_cols, categorical_cols) -> ColumnTransformer:
    bins_cols = ["age", "day"]
    addon_bins_cols = ["balance"]
    final_numerical_cols = [col for col in numerical_cols if col not in bins_cols and col not in addon_bins_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("bin_num", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform"), bins_cols),
            ("addon_bin_num", KBinsDiscretizer(n_bins=1000, encode="ordinal", strategy="uniform"), addon_bins_cols),
            ("num", StandardScaler(), final_numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ]
    )

    return preprocessor
