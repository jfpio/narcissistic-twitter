import os
from pathlib import Path

from dotenv import load_dotenv
from gspread_pandas import conf, Spread
import pandas as pd

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_gspread_config():
    credentials_json_path = os.getenv("GOOGLE_API_CREDENTIALS_JSON_PATH")
    return conf.get_config(conf_dir=str(Path(credentials_json_path).parent), file_name=Path(credentials_json_path).name)


def get_spread(file_url: str) -> Spread:
    return Spread(file_url, config=get_gspread_config())


def load_single_sheet_from_file(file_url: str, sheet_name: str) -> pd.DataFrame:
    spread = get_spread(file_url)
    return spread.sheet_to_df(sheet=sheet_name, index=None)


def save_df_to_sheet(df: pd.DataFrame, file_url: str, sheet_name: str):
    spread = get_spread(file_url)
    spread.df_to_sheet(df, sheet=sheet_name, replace=True, index=False)
