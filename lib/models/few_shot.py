from pathlib import Path
import re

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
from torch.nn import HuberLoss

from lib.models.abstract_base import AbstractBaseModel
from lib.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FewShotLearningModel(AbstractBaseModel):
    def __init__(self, model_name: str, embedding_model: str, api_key: str, number_of_shots: int, model_role: str):
        self.model_name = model_name
        self.number_of_shots = number_of_shots
        self.model_role = model_role
        self.chat_model = ChatOpenAI(model=self.model_name, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.vectorstore = None
        self.semantic_similarity_example_selector = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        examples = [{"post": x_example, "narcissism": str(y_example)} for x_example, y_example in zip(X, y)]
        to_vectorize = [" ".join(example.values()) for example in examples]

        self.vectorstore = Chroma.from_texts(to_vectorize, self.embeddings, metadatas=examples)
        self.semantic_similarity_example_selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore, k=self.number_of_shots
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=["input"],
            example_selector=self.semantic_similarity_example_selector,
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    ("human", "{post}"),
                    ("ai", "narcissism: {narcissism}"),
                ]
            ),
        )
        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.model_role),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, delta_huber: float = 1.0, quantile: float = 0.5) -> dict:
        valid_indices = np.logical_and(~np.isnan(y_true), ~np.isnan(y_pred))
        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]

        rmse = root_mean_squared_error(y_true_valid, y_pred_valid)
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        # Calculate the Maximal Absolute Error
        abs_errors = np.abs(y_true_valid - y_pred_valid)
        max_abs_error = np.max(abs_errors)

        huber_loss = HuberLoss(delta=delta_huber)
        huber_loss_value = huber_loss(torch.tensor(y_pred_valid), torch.tensor(y_true_valid)).item()

        quantile_loss = self.quantile_loss(torch.tensor(y_true_valid), torch.tensor(y_pred_valid), quantile).item()

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        chain = self.final_prompt | self.chat_model
        y_messages = [chain.invoke({"input": x}) for x in X]
        y_preds = [self.extract_response(y_message.content) for y_message in y_messages]
        return np.array(y_preds)

    def extract_response(self, content: str) -> float:
        match = re.search(r"\d+\.\d+", content)
        return float(match.group()) if match else np.nan

    def save(self, path: Path) -> None:
        # Implement if needed
        pass

    def load(self, path: Path) -> None:
        # Implement if needed
        pass
