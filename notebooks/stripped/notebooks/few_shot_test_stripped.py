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

from dotenv import load_dotenv

load_dotenv()


"""
# Example
"""

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=os.getenv('OPENAI_API_KEY'))
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "bread"})


"""
# Parameters
"""

post_type = 'post_travel' # 'post_travel' or 'post_abortion'
narcism_type = 'adm' # 'riv' or 'adm'
model_used = "gpt-4-1106-preview"
iterations = 10
number_of_shots = 5 # somewhere between 3 and 10
model_role = "You are a psychologist and you are assessing a patient's Narcissism. The patient is talking about their recent travel. Return only float number between 1 and 6."
train_path = "../data/split/train.csv"
validate_path = "../data/split/validate.csv"


"""
# Code
"""

"""
Here we used the most basic implementation, there is also option to use Dynamic few-shot prompting, but to my knowledge is not needed is this context as we have only one type of posts.
"""

# Get split data using pandas
df = pd.read_csv(train_path)

# Get the dictionary of the first x posts
example = df[[post_type,narcism_type]].iloc[0:number_of_shots]

example = example.to_dict(orient='records')

# Change the value name
for i in range(len(example)):
    example[i]['post'] = example[i].pop(post_type)
    example[i]['narcissism'] = example[i].pop(narcism_type)

example


model = ChatOpenAI(model=model_used, openai_api_key=os.getenv('OPENAI_API_KEY'))


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

test = df[[post_type,narcism_type]].iloc[4] # Test on train dataset
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


"""
# Analyze the results
"""

r = ai_message.content
match = re.search(r'\d+\.\d+', r)
if match:
    response = float(match.group())
else:
    response = None
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
        if float_number is not None:
            return float_number
        else:
            print(f"Wrong input: {text}")
    else:
        return None

# get the response
def get_response(final_prompt, model, input):
    chain = final_prompt | model
    ai_message = chain.invoke({"input": input})
    response = ai_message.content
    return response

# get the mean squared error
def get_mse(y_pred, y_true):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    return mse


"""
Add Neptune experiment observation
"""

# Run the functions
run = neptune.init_run(project = os.getenv('NEPTUNE_PROJECT'),
                       api_token = os.getenv('NEPTUNE_API_TOKEN'),
                       source_files=["few_shot_test.ipynb"],
                       tags=["few-shot", narcism_type, post_type])

run["type"] = "Few-shot learning"
params = {
    "model": model_used,
    "narc_type": narcism_type,
    "post_type": post_type,
    "prompt": model_role,
    "shots": number_of_shots
}
run["model/parameters"] = params # Save the parameters

y_pred = []
y_true = []

test_df = pd.read_csv(validate_path)
testset = test_df[[post_type,narcism_type]]

problems = []

for i in range(test.shape[0]):
    example = get_random_x_posts(train_path, post_type, narcism_type, number_of_shots) # Get some random examples
    few_shot_prompt = create_few_shot_prompt(example) 
    #input_dic = get_random_x_posts(validate_path, post_type, narcism_type, 1)
    #input = input_dic.pop(0)
    input = testset.iloc[i]
    final_prompt = create_final_prompt(few_shot_prompt,model_role)
    response_str = get_response(final_prompt, model, input.get(post_type))
    response = get_float(response_str)

    if response is not None: # Check if the model returned a number
        y_pred.append(response)
        y_true.append(input.get(narcism_type))
    else: # Else save the prompt that caused the error
        row_to_add = {'post': input.get(post_type), 'post_type': post_type, 'model_role': model_role, 'date': pd.Timestamp.now()}
        problems.append(row_to_add)


mse = get_mse(y_pred, y_true) # Calculate the mean squared error
print(mse)
run["mse"] = mse
run.stop() # Stop the run


# Make a dataframe of the problems
problems_df = pd.DataFrame(problems)
problems_df.to_csv("../data/responses/few_shot.csv", mode='a', index=False, header=False)

problems




