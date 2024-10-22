from pathlib import Path
import re
from typing import Union

import pandas as pd


def draw_data_for_human_as_model_ai(
    file_to_draw: Union[str, Path],
    file_to_get_size: Union[str, Path],
    target_name: Union[str, Path],
    remove_observation: bool = False,
    random_state=47,
) -> None:
    # Load the data test and train
    file_to_draw = Path(file_to_draw)
    file_to_get_size = Path(file_to_get_size)
    target_name = Path(target_name)

    data = pd.read_csv(file_to_draw)
    size_file = pd.read_csv(file_to_get_size)

    # Drop null values in the train and test set column 'post_ai'
    data = data.dropna(subset=["post_ai"])

    # Get a subset of train in the size of old_train
    data_for_human = data.sample(n=size_file.shape[0], random_state=random_state)

    # Drop all columns except 'post_ai', 'ADM', 'RIV'
    data_for_human = data_for_human[["post_ai", "adm", "riv"]]

    # Remove observations
    if remove_observation:
        data_for_human[["adm", "riv"]] = ["", ""]

    # save the data
    data_for_human.to_csv(target_name, index=False)

    match = re.search(r"/([^/]+)\.", str(target_name))
    if match:
        target = match.group(1)
    else:
        target = target_name

    print(f"{target} shape: {data_for_human.shape}")


if __name__ == "__main__":
    draw_data_for_human_as_model_ai(
        file_to_draw="data/split/full_train.csv",
        file_to_get_size="data/split/train.csv",
        target_name="data/split/human_train_ai.csv",
        remove_observation=False,
        random_state=47,
    )

    draw_data_for_human_as_model_ai(
        file_to_draw="data/split/full_test.csv",
        file_to_get_size="data/split/test.csv",
        target_name="data/split/human_test_ai.csv",
        remove_observation=True,
        random_state=47,
    )
