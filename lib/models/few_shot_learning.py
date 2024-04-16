from pathlib import Path
import re

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np

from lib.models.abstract_base import BaseModel


class FewShotLearningModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, number_of_shots: int, model_role: str):
        self.model_name = model_name
        self.api_key = api_key
        self.number_of_shots = number_of_shots
        self.model_role = model_role
        self.chat_model = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key)
        self.final_prompt_template = None  # Placeholder for the prompt created during training

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        'Trains' the model by preparing the few-shot prompt using example data provided.
        This step simulates training by setting up the model configuration for use during prediction.

        Parameters:
            X (np.ndarray): Array of posts as strings.
            y (np.ndarray): Array of corresponding narcissism scores as floats.
        """
        if len(X) < self.number_of_shots:
            raise ValueError("Not enough data provided for the number of requested shots.")
        few_shot_prompt = self.create_few_shot_prompt(X[: self.number_of_shots], y[: self.number_of_shots])
        self.final_prompt_template = self.create_final_prompt(few_shot_prompt)

        if self.final_prompt_template is None:
            raise Exception("Failed to create final prompt template")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions by using the final prompt template and invoking the model.
        Assumes that `train` has been called beforehand to set up the final prompt.
        """
        if self.final_prompt_template is None:
            raise ValueError("Model has not been trained. Please call the train method before predicting.")

        predictions = []
        for input_text in X:
            ai_message = self.chat_model.invoke({"input": self.final_prompt_template.format(input=input_text)})
            response = self.extract_response(ai_message.content)
            predictions.append(response)
        return np.array(predictions)

    def create_few_shot_prompt(
        self, posts: np.ndarray, narcissism_scores: np.ndarray
    ) -> FewShotChatMessagePromptTemplate:
        examples = [{"post": post, "narcissism": score} for post, score in zip(posts, narcissism_scores)]
        example_prompt = ChatPromptTemplate.from_messages([("human", "{post}"), ("ai", "narcissism: {narcissism}")])
        return FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

    def create_final_prompt(self, few_shot_prompt: FewShotChatMessagePromptTemplate) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([("system", self.model_role), few_shot_prompt, ("human", "{input}")])

    def extract_response(self, content: str) -> float:
        match = re.search(r"\d+\.\d+", content)
        return float(match.group()) if match else None

    def save(self, path: Path) -> None:
        # Implement if needed
        pass

    def load(self, path: Path) -> None:
        # Implement if needed
        pass
