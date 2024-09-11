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
    remove_obsolete: bool = True,
    append: bool = False,
) -> None:
    file_to_open = Path(file_to_open)
    train_target_name = Path(train_target_name)
    test_target_name = Path(test_target_name)

    data = pd.read_csv(file_to_open)

    test_size = 0.25 if append else 0.50

    train_data, rest_data = train_test_split(data, test_size=test_size, random_state=random_state)
    test_data, validate_data = train_test_split(rest_data, test_size=0.50, random_state=random_state)

    if remove_obsolete:
        # Save the original indices
        train_indices = train_data.index
        test_indices = test_data.index
        validate_indices = validate_data.index

        # Remove the observations that should not be in the dataset
        updated_data = remove_obsolete_observations(data=data)

        # Filter valid indices
        valid_train_indices = train_indices.intersection(updated_data.index)
        valid_test_indices = test_indices.intersection(updated_data.index)
        valid_validate_indices = validate_indices.intersection(updated_data.index)

        train_data = updated_data.loc[valid_train_indices]
        test_data = updated_data.loc[valid_test_indices]
        validate_data = updated_data.loc[valid_validate_indices]

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validate_data.reset_index(drop=True, inplace=True)

    if append:
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True

    train_data.to_csv(train_target_name, mode=mode, header=header, index=False)
    test_data.to_csv(test_target_name, mode=mode, header=header, index=False)
    validate_data.to_csv(validate_target_name, mode=mode, header=header, index=False)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Validation data shape: {validate_data.shape}")


def remove_obsolete_observations(data: pd.DataFrame) -> pd.DataFrame:
    # This function removes the obsolete observations from the dataset
    # It is implemented in that way to maintain the consistency with splitting the dataset

    indices_to_delete = []
    indices_to_delete.extend(
        data[
            data["post_travel"]
            == (
                "With twitter/X you can create threads and using that you can "
                "engage with the followers. I could describe my travel using "
                "hashtags with the travel destination and what not"
            )
        ].index
    )
    indices_to_delete.extend(
        data[
            data["post_travel"]
            == (
                "I am unlikely to share much in the way for thoughts, "
                "reflections or feelings about my travels. I *may* post "
                "a photo with a caption stating where it was taken. "
                'I may even add such a superlative as "lovely, or sunny" or somesuch'
            )
        ].index
    )
    indices_to_delete.extend(
        data[
            data["post_abortion"]
            == (
                "A lot of countries have already banned abortions so "
                "if I was to make a post then it would be a pro and con situation"
            )
        ].index
    )
    indices_to_delete.extend(data[data["post_abortion"] == ("I do not have any thoughts on the above.")].index)
    indices_to_delete.extend(
        data[
            data["post_abortion"]
            == (
                "I would be very unlikely to post about this. "
                "Althought I would be extremely against this. "
                "I would be much more likely to RT someone else."
            )
        ].index
    )
    indices_to_delete.extend(
        data[data["post_abortion"] == ("This is something I would prefer not to comment on")].index
    )
    updated_data = data.drop(indices_to_delete)
    return updated_data


if __name__ == "__main__":
    split_datasets(
        "data/processed/processed_new_data.csv",
        "data/split/full_train.csv",
        "data/split/full_test.csv",
        "data/split/full_validate.csv",
        random_state=47,
        remove_obsolete=False,
        append=False,
    )

    split_datasets(
        "data/processed/processed_data.csv",
        "data/split/full_train.csv",
        "data/split/full_test.csv",
        "data/split/full_validate.csv",
        random_state=47,
        remove_obsolete=True,
        append=True,
    )
