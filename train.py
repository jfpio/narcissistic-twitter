from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from lib.datamodules.datamodule import NarcissisticPostsSimpleDataModule
from lib.models.abstract_base import BaseModel
from lib.utils import extras, get_metric_value, instantiate_loggers, RankedLogger, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: NarcissisticPostsSimpleDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))[0]

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
    }

    # log cfg with logger
    logger.log_hyperparams(cfg)

    if cfg.get("train"):
        log.info("Starting training!")
        model.train(datamodule.data_train.posts, datamodule.data_train.labels)

    train_metrics = model.evaluate(model.predict(datamodule.data_train.posts), datamodule.data_train.labels)
    val_metrics = model.evaluate(model.predict(datamodule.data_train.posts), datamodule.data_train.labels)

    log.info(f"Train metrics: {train_metrics}")
    log.info(f"Validation metrics: {val_metrics}")

    if cfg.get("test"):
        log.info("Starting testing!")
        test_metrics = model.evaluate(model.predict(datamodule.data_train.posts), datamodule.data_train.labels)
        test_metrics_second_category = model.evaluate(
            model.predict(datamodule.data_test_second_category.posts), datamodule.data_test_second_category.labels
        )

    log.info(f"Test metrics: {test_metrics}")
    metric_dict = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_second_category": test_metrics_second_category,
    }

    logger.log_metrics(metric_dict)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
