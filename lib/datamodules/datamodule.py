import pandas as pd

from lib.datamodules.dataset import NarcissisticPostDataset


class NarcissisticPostsSimpleDataModule:
    def __init__(
        self,
        data_dir: str = "data/",
        train_file: str = "train.csv",
        val_file: str = "val.csv",
        test_file: str = "test.csv",
        post_category: str = "post_travel",
        second_post_category: str = "post_abortion",
        label_category: str = "adm",
    ) -> None:
        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.post_category = post_category
        self.second_post_category = second_post_category
        self.label_category = label_category

        self.setup()

    def setup(self) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Load data from files
            train_df = pd.read_csv(self.data_dir + self.train_file)
            val_df = pd.read_csv(self.data_dir + self.val_file)
            test_df = pd.read_csv(self.data_dir + self.test_file)

            self.data_train = NarcissisticPostDataset(
                train_df[self.post_category],
                train_df[self.label_category],
            )
            self.data_val = NarcissisticPostDataset(
                val_df[self.post_category],
                val_df[self.label_category],
            )
            self.data_test = NarcissisticPostDataset(
                test_df[self.post_category],
                test_df[self.label_category],
            )

            self.data_test_second_category = NarcissisticPostDataset(
                test_df[self.second_post_category],
                test_df[self.label_category],
            )
