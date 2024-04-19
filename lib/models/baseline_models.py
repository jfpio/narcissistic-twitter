from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluates the model using mean squared error.
        :param y_true: True labels or target values.
        :param y_pred: Predictions made by the model.
        :return: A dictionary containing the 'mse' metric.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        root_mse = np.sqrt(mse)

        return {"mse": mse, "r2_score": r2, "root_mse": root_mse}

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
    def __init__(self, max_iter=1000):
        self.model = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("regressor", MLPRegressor(max_iter=max_iter)),
            ]
        )


class SVRModel(BaselineMLModel):
    def __init__(self):
        self.model = Pipeline([("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("svr", SVR())])


class RandomForestRegressorModel(BaselineMLModel):
    def __init__(self):
        self.model = Pipeline(
            [("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("rfr", RandomForestRegressor())]
        )


class DecisionTreeRegressorModel(BaselineMLModel):
    def __init__(self):
        self.model = Pipeline(
            [("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("dtr", DecisionTreeRegressor())]
        )


class GradientBoostingRegressorModel(BaselineMLModel):
    def __init__(self):
        self.model = Pipeline(
            [("vectorizer", CountVectorizer()), ("tfidf", TfidfTransformer()), ("gbr", GradientBoostingRegressor())]
        )
