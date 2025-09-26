import joblib
import pandas as pd

from pathlib import Path
from typing import Any


class HelperError(Exception):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


# ----- Functions ----- #
def read_kaggle_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    print("--- Dataset Overview ---")
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset columns: {data.columns}")
    print(data.info())

    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if not missing_data.empty:
        print(f"Missing values in dataset:{missing_data}")

    print(f"Dataset sample:\n{data.head()}")

    return data


def save_kaggle_result(data: pd.DataFrame, file_name: str, save_path: Path, target_cols: str = "y") -> None:
    if "id" not in data.columns:
        raise HelperError("ID missing in dataframe. Requires for Kaggle submission")

    if target_cols not in data.columns:
        raise HelperError(f"{target_cols} missing in dataframe. Requires for Kaggle Submission")

    data.to_csv(save_path / file_name)


def save_model(model_card: dict[str, Any], model_name: str, save_path: Path) -> None:
    joblib.dump(value=model_card, filename=save_path / model_name)
    print(f"Model ({model_name}) saved.")
