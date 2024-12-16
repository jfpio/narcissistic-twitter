from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import torch
from torch.nn import HuberLoss

from lib.models.abstract_base import AbstractBaseModel


class BaselineMLModel(AbstractBaseModel):
    def __init__(self, model_pipeline: Pipeline):
        """
        Initializes the BaselineMLModel with a specific Scikit-learn pipeline.
        :param model_pipeline: A scikit-learn pipeline that includes preprocessing and a learning algorithm.
        """
        self.model = model_pipeline

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model using the provided data.
        :param X: Feature matrix as a numpy array.
        :param y: Labels or target values as a numpy array.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values using the trained model for the provided feature matrix.
        :param X: Feature matrix as a numpy array.
        :return: Predictions as a numpy array.
        """
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, delta_huber: float = 1.0, quantile: float = 0.5) -> dict:
        """
        Evaluates the model using mean squared error.
        :param y_true: True labels or target values.
        :param y_pred: Predictions made by the model.
        :return: A dictionary containing the 'mse' metric.
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        # Calculate the Maximal Absolute Error
        abs_errors = np.abs(y_true - y_pred)
        max_abs_error = np.max(abs_errors)

        huber_loss = HuberLoss(delta=delta_huber)
        huber_loss_value = huber_loss(torch.tensor(y_pred), torch.tensor(y_true)).item()

        quantile_loss = self.quantile_loss(torch.tensor(y_true), torch.tensor(y_pred), quantile).item()

        return {
            "rmse": rmse,
            "mae": mae,
            "maxAE": max_abs_error,
            "huber_loss": huber_loss_value,
            "quantile_loss": quantile_loss,
        }

    def quantile_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float = 1.0) -> torch.Tensor:
        error = y_true - y_pred
        return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class LinearRegressionModel(BaselineMLModel):
    def __init__(self):
        self.model = Pipeline(
            [("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("regressor", LinearRegression())]
        )


class MLPRegressorModel(BaselineMLModel):
    def __init__(
        self,
        max_iter: int,
        activation: Literal["relu", "identity", "logistic", "tanh"],
        solver: Literal["lbfgs", "sgd", "adam"],
    ):
        self.model = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("regressor", MLPRegressor(max_iter=max_iter, activation=activation, solver=solver)),
            ]
        )


class SVRModel(BaselineMLModel):
    def __init__(self, kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]):
        self.model = Pipeline(
            [("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("svr", SVR(kernel=kernel))]
        )


class RandomForestRegressorModel(BaselineMLModel):
    def __init__(self, criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]):
        self.model = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("rfr", RandomForestRegressor(criterion=criterion)),
            ]
        )


class DecisionTreeRegressorModel(BaselineMLModel):
    def __init__(self, criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"]):
        self.model = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("dtr", DecisionTreeRegressor(criterion=criterion)),
            ]
        )


class GradientBoostingRegressorModel(BaselineMLModel):
    def __init__(self, loss: Literal["squared_error", "absolute_error", "huber", "quantile"]):
        self.model = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("gbr", GradientBoostingRegressor(loss=loss)),
            ]
        )
