import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split


# Load the data
# Data path
data_path = 'data/teach AI 12.12.23.csv'
data = pd.read_csv(data_path)
data.head()


# Analyze the data
# Check the data types
# Check the missing values
data.replace('#NULL!', np.nan, inplace=True)
data.isnull().sum()


# Get columns
data.columns


# List of not needed columns to drop:
columns_to_drop = ['ID','CNI_check','GCB_check_','Progress','Finished','Location']
data.drop(columns=columns_to_drop, inplace=True)
data.head(2)


def clean(review):
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() 
                       if word not in stopwords.words('english')])
    return review


data['POST___ABORTION'] = data['POST___ABORTION'].apply(clean)
data['POST___TRAVEL'] = data['POST___TRAVEL'].apply(clean)
data.head(2)


# Divide the data for validation and testing
# Split the data into training and testing sets
X = data['POST___ABORTION']
y = data['POST___TRAVEL']
train, validate = np.split(data.sample(frac=1,random_state=47), [int(.8*len(data))])

train.reset_index(drop=True, inplace=True)
validate.reset_index(drop=True, inplace=True)


# check the shape of the data
print(train.shape)
print(validate.shape)


# TODO: Remove the most common words from the data (as abortion, goverment,
#       women, right, choice, amazing, time, beautiful, etc.)


# Save the data
train.to_csv('data/processed_data.csv', index=False)
validate.to_csv('data/processed_data_validate.csv', index=False)




