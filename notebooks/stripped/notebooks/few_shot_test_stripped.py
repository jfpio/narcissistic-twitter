from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
import os

from sklearn.metrics import mean_squared_error 

import pandas as pd
import numpy as np
import re
import neptune


"""
# Example
"""

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key="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")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "bread"})


"""
# Parameters
"""

post_type = 'post_travel'
narcism_type = 'adm'
model_used = "gpt-3.5-turbo-1106"
iterations = 10
number_of_shots = 5 # somewhere between 3 and 10
model_role = "You are a psychologist and you are assessing a patient's Narcissism. The patient is talking about their recent travel. Return only float number between 1 and 6."
train_path = "../data/split/train_data.csv"
validate_path = "../data/split/validate_data.csv"


"""
# Code
"""

"""
Here we used the most basic implementation, there is also option to use Dynamic few-shot prompting, but to my knowledge is not needed is this context as we have only one type of posts.
"""

# Get split data using pandas
path = "../data/split/train_data.csv"
df = pd.read_csv(path)

# Get the dictionary of the first x posts
example = df[[post_type,narcism_type]].iloc[0:number_of_posts]

example = example.to_dict(orient='records')

# Change the value name
for i in range(len(example)):
    example[i]['post'] = example[i].pop(post_type)
    example[i]['narcissism'] = example[i].pop(narcism_type)

example


print(os.environ.get('OPENAI_API_KEY'))


# TODO - Load the API key from the environment
model = ChatOpenAI(model=model_used, openai_api_key="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")


# This is a prompt template used to format each example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{post}"),
        ("ai", "result: {narcissism}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=example,
)

# print(few_shot_prompt.format())


"""
#### Use train and validate dataset!!!
"""

test = df[[post_type,narcism_type]].iloc[4]
input = test.iloc[0]
input


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return a narcissism number between 1 and 6."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

final_prompt


chain = final_prompt | model

ai_message = chain.invoke({"input": input})

ai_message


r = get_float(ai_message.content)
r


"""
# Analyze the results
"""

response = ai_message.content

response


test.iloc[1]


y_pred = []
y_true = []
y_pred.append(response)
y_true.append(test.iloc[1])


mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
mse


"""
# Implementation
"""

# functions

# get random x posts
def get_random_x_posts(path, post_type, narcism_type, number_of_posts):

    df = pd.read_csv(path)
    example = df[[post_type,narcism_type]].sample(number_of_posts)
    example = example.to_dict(orient='records')

    # Change the value name
    for i in range(len(example)):
        example[i]['post'] = example[i].pop(post_type)
        example[i]['narcissism'] = example[i].pop(narcism_type)

    return example

# create a few shot prompt
def create_few_shot_prompt(example):
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{post}"),
            ("ai", "narcissism: {narcissism}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=example,
    )
    return few_shot_prompt

# create a final prompt
def create_final_prompt(few_shot_prompt,model_role):
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", model_role),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt

# get float number from a string
def get_float(text):
    # Use regular expression to find numerical value
    match = re.search(r'\d+\.\d+', text)
    if match:
        float_number = float(match.group())
        return float_number
    else:
        return None

# get the response
def get_response(final_prompt, model, input):
    chain = final_prompt | model
    ai_message = chain.invoke({"input": input})
    response = ai_message.content
    print(response)
    # get a float number from a string
    response = get_float(response)
    return response

# get the mean squared error
def get_mse(y_pred, y_true):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    return mse


"""
Add Neptune experiment observation
"""

# Run the functions
# TODO: Set API token in the environment variable
run = neptune.init_run(project = "NarcisissticTwitter/Twitter",
                       api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYmEzZjU5ZS1kZDIzLTQwNTEtYjQ4Ni1hYTlhMTFjY2YzMjIifQ==")
# TODO: Add logging of input posts (and responses?) to Neptune
run["algorithm"] = "Few-shot learning"
params = {
    "model": model_used,
    "narc_type": narcism_type,
    "post_type": post_type,
    "prompt": model_role,
    "shots": number_of_shots
}
run["model/parameters"] = params
run.add_tags(["few-shot", "narcissism", narcism_type])
y_pred = []
y_true = []
for i in range(iterations):
    example = get_random_x_posts(train_path, post_type, narcism_type, number_of_shots)
    few_shot_prompt = create_few_shot_prompt(example)
    input_dic = get_random_x_posts(validate_path, post_type, narcism_type, 1)
    input = input_dic.pop(0)
    final_prompt = create_final_prompt(few_shot_prompt,model_role)
    response = get_response(final_prompt, model, input.get('post'))
    y_pred.append(response)
    y_true.append(input.get('narcissism'))

mse = get_mse(y_pred, y_true)
print(mse)
run["mse"] = mse
run.stop()


