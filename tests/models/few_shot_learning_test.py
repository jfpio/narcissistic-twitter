from unittest.mock import MagicMock

from langchain_openai import ChatOpenAI
from numpy import array
import pytest

from lib.models.few_shot_learning import FewShotLearningModel


@pytest.fixture
def mock_chat_model(monkeypatch):
    # Mock the ChatOpenAI to avoid actual API calls
    mock = MagicMock(spec=ChatOpenAI)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock)
    return mock


@pytest.fixture
def few_shot_model(mock_chat_model):
    # Instantiate the model with a mock ChatOpenAI
    model = FewShotLearningModel(
        model_name="gpt-3.5-turbo-1106",
        api_key="dummy_key",
        number_of_shots=3,
        model_role="Assess the narcissism level.",
    )
    model.chat_model = mock_chat_model
    return model


def test_extract_response():
    content = "The narcissism score is 4.5."
    model = FewShotLearningModel(
        model_name="gpt-3.5-turbo-1106",
        api_key="dummy_key",
        number_of_shots=3,
        model_role="Assess the narcissism level.",
    )
    assert model.extract_response(content) == 4.5, "The extracted score should be 4.5"


def test_extract_response_no_match():
    content = "No numerical score mentioned."
    model = FewShotLearningModel(
        model_name="gpt-3.5-turbo-1106",
        api_key="dummy_key",
        number_of_shots=3,
        model_role="Assess the narcissism level.",
    )
    assert model.extract_response(content) is None, "No score should be extracted"


def test_predict_without_train(few_shot_model, mock_chat_model):
    posts = array(["New post for testing"])
    with pytest.raises(ValueError, match="Model has not been trained. Please call the train method before predicting."):
        few_shot_model.predict(posts)


def test_successful_train_and_predict(few_shot_model, mock_chat_model):
    posts = array(["New post for testing", "Second post", "Third post"])
    scores = array([2.5, 3.5, 4.5])
    few_shot_model.train(posts, scores)
    mock_chat_model.invoke.return_value = MagicMock(content="The narcissism score is 3.2.")

    prediction = few_shot_model.predict(posts[:1])
    assert len(prediction) == 1, "Should return one prediction"
    assert prediction[0] == 3.2, "The prediction should match the mocked response"
