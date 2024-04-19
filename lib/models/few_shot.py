from pathlib import Path
import re

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from lib.models.abstract_base import AbstractBaseModel
from lib.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FewShotLearningModel(AbstractBaseModel):
    def __init__(self, model_name: str, api_key: str, number_of_shots: int, model_role: str):
        self.model_name = model_name
        self.number_of_shots = number_of_shots
        self.model_role = model_role
        self.chat_model = ChatOpenAI(model=self.model_name, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings()

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

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        root_mse = np.sqrt(mse)
        return {"mse": mse, "r2_score": r2, "root_mse": root_mse}

    def predict(self, X: np.ndarray) -> np.ndarray:
        chain = self.final_prompt | self.chat_model
        y_messages = [chain.invoke({"input": x}) for x in X]
        y_preds = [self.extract_response(y_message.content) for y_message in y_messages]
        return np.array(y_preds)

    def extract_response(self, content: str) -> float:
        match = re.search(r"\d+\.\d+", content)
        return float(match.group()) if match else None

    def save(self, path: Path) -> None:
        # Implement if needed
        pass

    def load(self, path: Path) -> None:
        # Implement if needed
        pass
