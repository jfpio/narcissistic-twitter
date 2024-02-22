import hashlib
import io
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional
import zipfile

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import yaml

load_dotenv()


def get_credentials() -> service_account.Credentials:
    credentials_json_path = os.getenv("GOOGLE_API_CREDENTIALS_JSON_PATH")
    creds = service_account.Credentials.from_service_account_file(credentials_json_path)
    return creds


def upload_file_to_google_drive(file_path: Path, file_name: str, parent_folder_id: str) -> str:
    service = build("drive", "v3", credentials=get_credentials())

    file_metadata = {"name": file_name, "parents": [parent_folder_id]}
    media = MediaFileUpload(str(file_path), resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id", uploadType="resumable").execute()
    file_id = file.get("id")
    logging.info(f"File ID: {file_id} uploaded to Google Drive.")
    return file_id


def create_config_for_data(data_name: str, file_id: str, cascade: bool = False) -> None:
    if cascade:
        target = "lib.datamodule.dataset.CascadeDataset.load_from_dir"
    else:
        target = "lib.datamodule.dataset.TweetsNewsDataset.load_from_dir"
    directory = "${paths.data_dir}/" + data_name
    google_url = f"https://drive.google.com/file/d/{file_id}"
    dict_file = {"_target_": target, "directory": directory, "google_url": google_url}
    with open(f"configs/data/{data_name}.yaml", "w") as file:
        yaml.dump(dict_file, file)


def zip_dir_and_upload_to_google_drive(
    path: Path, google_drive_folder_id: str, save_config: bool = False, cascade: bool = False
):
    filename = shutil.make_archive(base_name=path.name, format="zip", root_dir=str(path))
    zip_path = Path(filename)
    file_id = upload_file_to_google_drive(zip_path, zip_path.name, google_drive_folder_id)
    if save_config:
        create_config_for_data(path.name, file_id, cascade)
        logging.info(f"Created hydra config file for {path.name}.")
    zip_path.unlink()


def download_and_save_file_from_google_drive(
    file_url: str,
    root: str,
    filename: str,
    md5: Optional[str] = None,
):
    """
    Based on
    - https://developers.google.com/drive/api/guides/manage-downloads#python
    - https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L206
    """
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)
    file_id = file_url.split("/")[-1]
    os.makedirs(root, exist_ok=True)

    if _check_integrity(fpath, md5):
        logging.info(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    service = build("drive", "v3", credentials=get_credentials())

    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        logging.info(f"Download {int(status.progress() * 100)}.")

    if zipfile.is_zipfile(file) and not filename.endswith(".zip"):
        fpath = fpath + ".zip"

    with open(fpath, "wb") as f:
        f.write(file.getbuffer())

    if md5 and not _check_md5(fpath, md5):
        raise RuntimeError(
            f"The MD5 checksum of the download file {fpath} does not match the one on record."
            f"Please delete the file and try again. "
            f"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues."
        )


def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py"""
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    """Taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py"""
    return md5 == _calculate_md5(fpath, **kwargs)


def _check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    """Taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py"""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return _check_md5(fpath, md5)
