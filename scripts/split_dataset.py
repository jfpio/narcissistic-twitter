from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


def split_datasets(
    file_to_open: Union[str, Path],
    train_target_name: Union[str, Path],
    test_target_name: Union[str, Path],
    validate_target_name: Union[str, Path],
    random_state: int = 47,
) -> None:
    file_to_open = Path(file_to_open)
    train_target_name = Path(train_target_name)
    test_target_name = Path(test_target_name)

    data = pd.read_csv(file_to_open)

    train_data, rest_data = train_test_split(data, test_size=0.50, random_state=random_state)
    test_data, validate_data = train_test_split(rest_data, test_size=0.50, random_state=random_state)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validate_data.reset_index(drop=True, inplace=True)

    train_data.to_csv(train_target_name, index=False)
    test_data.to_csv(test_target_name, index=False)
    validate_data.to_csv(validate_target_name, index=False)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Validation data shape: {validate_data.shape}")


if __name__ == "__main__":
    split_datasets(
        "data/processed/processed_data.csv",
        "data/processed/processed_data_train.csv",
        "data/processed/processed_data_test.csv",
        "data/processed/processed_data_validate.csv",
        random_state=47,
    )
