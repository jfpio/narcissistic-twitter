from typing import Any, Dict, Tuple

from lightning import LightningModule
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
from torch.nn import HuberLoss
from torchmetrics import MeanSquaredError
from transformers import BertModel


class NarcissisticPostBERTLitModule(LightningModule):
    def __init__(
        self,
        hg_bert_model_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = None
        self.classifier_head = None

        # for averaging loss across batches
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()

    def forward(self, x: dict) -> torch.Tensor:
        # Get the pooled output from BERT
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_head(pooled_output)

        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        y = batch["labels"]
        preds = self(x)

        loss = torch.nn.functional.mse_loss(preds.squeeze(), y)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(preds.view(-1), targets)
        self.log("train/mse", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics, val loss is mean squared error
        self.val_loss(preds.squeeze(), targets)
        self.log("val/mse", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(preds.squeeze(), targets)
        self.log("test/mse", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "test/root_mse",
            root_mean_squared_error(targets.cpu(), preds.squeeze().cpu()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/mae",
            mean_absolute_error(targets.cpu(), preds.squeeze().cpu()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/maxAE",
            torch.max(torch.abs(targets.cpu() - preds.squeeze().cpu())),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # TODO: Move the HuberLoss to init? or is there a better solution?
        huber_loss = HuberLoss(delta=1.0)  # TODO: make delta a hyperparameter

        self.log(
            "test/HuberLoss",
            huber_loss(targets.cpu(), preds.squeeze().cpu()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/quantile_loss",
            self.quantile_loss(targets.cpu(), preds.squeeze().cpu(), 1.0).item(),  # TODO: implement quantile value
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def quantile_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float = 1.0) -> torch.Tensor:
        error = y_true - y_pred
        return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.encoder = BertModel.from_pretrained(self.hparams.hg_bert_model_name)
        self.dropout = torch.nn.Dropout(self.hparams.dropout_rate)
        self.classifier_head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mse",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
