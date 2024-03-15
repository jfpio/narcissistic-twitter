from typing import Any, Optional

from lightning import LightningDataModule
import pandas as pd
from torch.utils.data import DataLoader

from lib.data.dataset import NarcissisticPostDataset


class NarcissisticPostsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_file: str = "train.csv",
        val_file: str = "val.csv",
        test_file: str = "test.csv",
        post_category: str = "post_travel",
        label_category: str = "adm",
        tokenizer: Any = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) \
                    is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Load data from files
            train_df = pd.read_csv(self.hparams.data_dir + self.hparams.train_file)
            val_df = pd.read_csv(self.hparams.data_dir + self.hparams.val_file)
            test_df = pd.read_csv(self.hparams.data_dir + self.hparams.test_file)

            # Define tokenizer and max_token_len
            tokenizer = self.tokenizer
            max_token_len = self.hparams.max_token_len

            self.data_train = NarcissisticPostDataset(
                train_df[self.hparams.post_category],
                train_df[self.hparams.label_category],
                tokenizer,
                max_token_len
            )
            self.data_val = NarcissisticPostDataset(
                val_df[self.hparams.post_category],
                val_df[self.hparams.label_category],
                tokenizer,
                max_token_len
            )
            self.data_test = NarcissisticPostDataset(
                test_df[self.hparams.post_category],
                test_df[self.hparams.label_category],
                tokenizer,
                max_token_len
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    _ = NarcissisticPostsDataModule()
