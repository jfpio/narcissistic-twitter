from pathlib import Path
from typing import Union

import numpy as np
import pyreadstat


def process_raw_data(file_to_open: Union[str, Path], target_name: Union[str, Path]) -> None:
    # Ensure paths are in Path format for consistency
    file_to_open = Path(file_to_open)
    target_name = Path(target_name)

    # Read the data
    data, meta = pyreadstat.read_sav(file_to_open)

    ffni_columns = [col for col in data.columns if col.startswith("FFNI")]
    mean_columns = [col for col in data.columns if col.endswith("MEAN")]
    columns_to_drop = (
        ["ID", "CNI_check", "GCB_check_", "Progress", "Finished", "Location"] + ffni_columns + mean_columns
    )
    data.drop(columns=columns_to_drop, inplace=True)

    data.rename(
        columns={
            "Other_portals_1": "None",
            "Other_portals_2": "Facebook",
            "Other_portals_3": "Instagram",
            "Other_portals_4": "TikTok",
            "Other_portals_5": "LinkedIn",
            "Other_portals_6": "Pinterest",
            "Other_portals_7": "Other",
        },
        inplace=True,
    )

    # Change the column names convention to snake_case
    data.columns = [col.lower() for col in data.columns]

    # Remove double underscores
    data.columns = [col.replace("___", "_") for col in data.columns]

    data.replace("", np.nan, inplace=True)
    data.replace("nan", np.nan, inplace=True)

    # remove new line characters
    data = data.replace("\n", " ", regex=True)
    # remove double spaces
    data.replace("  ", " ", regex=True)
    # remove leading and trailing spaces
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert age to integer
    data["age"] = data["age"].astype(int)

    # Make ADM and RIV have only 3 values after the decimal point
    data["adm"] = data["adm"].round(3)
    data["riv"] = data["riv"].round(3)

    data.to_csv(target_name, index=False)


if __name__ == "__main__":
    process_raw_data("data/raw/teach AI 12.12.23.sav", "data/processed/processed_data.csv")
