from pathlib import Path

from lib.google_api.drive import download_and_save_file_from_google_drive


class FileOnGoogleDrive:
    def __init__(self, path: Path, file_url: str) -> None:
        """
        Initializes a FileOnGoogleDrive object.

        Args:
            path (Path): The local path where the file should be saved or accessed.
            file_url (str): The url of the file in Google Drive.
        """
        self.path = path
        self.file_url = file_url

    def get_path_and_download_if_path_not_exist(self) -> Path:
        """
        Returns the path of the file and downloads the file from Google Drive if it doesn't exist locally.
        """
        self.download_if_not_exist()
        return self.path

    def download_if_not_exist(self) -> None:
        """
        Downloads the file from Google Drive if it doesn't exist locally.
        """
        if not self.path.exists():
            download_and_save_file_from_google_drive(
                file_url=self.file_url,
                root=str(self.path.parent),
                filename=self.path.name,
            )
