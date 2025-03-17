from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from transformers import pipeline
import ast
import os

def generate_emotion_data(
    folder_to_open: Union[str, Path],
    target_name: Union[str, Path],
    model: str = "j-hartmann/emotion-english-distilroberta-base",
    top_k: int = None, # number from 1 to 7, None for all
) -> None:
    
    classifier = pipeline("text-classification", model=model, top_k=top_k)

    folder_to_open = Path(folder_to_open)

    files = os.listdir(folder_to_open)

    data = pd.DataFrame()

    for file in files:
        if data.empty:
            data = pd.read_csv(folder_to_open / file)
        else:
            data = pd.concat([data, pd.read_csv(folder_to_open / file)], axis=0)

    data = data.reset_index(drop=True)

    # add new column to data
    data["tr_emotion"] = data["post_travel"].apply(lambda x: classifier(x)[0])
    data["ab_emotion"] = data["post_abortion"].apply(lambda x: classifier(x)[0])
    data["ai_emotion"] = data["post_ai"].apply(lambda x: classifier(x)[0] if x is not np.nan else np.nan)

    # save data into reporting folder
    data.to_csv(target_name, index=False)

if __name__ == "__main__":
    generate_emotion_data(
        folder_to_open="data/processed",
        target_name="data/reporting/emotion_data.csv",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

