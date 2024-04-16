"""
# Human as a model

This is another way of baseline comparison when the psychologist is shown the same data as the model: post (travel) and two types of narcissism scores - the training dataset.

| post_tavel         | adm     | riv |
|--------------|-----------|------------|
| I wish I could travel 24/7 and get paid for it | 1.444 | 1.111 |

And then has to asses the admiration and rivalry scores in the test dataset based on shown posts.
"""

"""
## Load the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


path_to_human_assessed = '../data/raw/human_as_model_travel.csv'
path_to_test = '../data/split/test.csv'


# load the human assessed data
human_data = pd.read_csv(path_to_human_assessed)
human_data.head()


# load the test data
test_data = pd.read_csv(path_to_test)
test_data[['post_travel','adm','riv']].head()


# drop not needed columns
test_data = test_data[['post_travel','adm','riv']]


# Merge the two dataframes
merged_data = pd.merge(human_data, test_data, on='post_travel', suffixes=('_human', '_original'))
# Check the length of the merged data
print(f"Merged correctly: {len(merged_data)==len(human_data)==len(test_data)}")
merged_data.head()


"""
# Mean squared error for the human assessed data
"""

# Calculate the mean squared error
mse_adm = mean_squared_error(merged_data['adm_human'], merged_data['adm_original'])
mse_riv = mean_squared_error(merged_data['riv_human'], merged_data['riv_original'])
print(f"Mean Squared Error for adm: {mse_adm}")
print(f"Mean Squared Error for riv: {mse_riv}")


"""
# Other metrics
"""

# other metrics
# Calculate the mean absolute error
mae_adm = np.mean(np.abs(merged_data['adm_human'] - merged_data['adm_original']))
mae_riv = np.mean(np.abs(merged_data['riv_human'] - merged_data['riv_original']))
print(f"Mean Absolute Error for adm: {mae_adm}")
print(f"Mean Absolute Error for riv: {mae_riv}")


"""
### Distribution
"""

plt.figure(figsize=(10, 6))
ax1 = plt.subplot(1, 2, 1)
sns.histplot(merged_data[['adm_human']],x = 'adm_human', color='orange', edgecolor='red', kde=True)
plt.title('Human scores')
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
sns.histplot(merged_data[['adm_original']],x = 'adm_original', kde=True)
plt.title('True scores')
plt.show()


plt.figure(figsize=(10, 6))
ax1 = plt.subplot(1, 2, 1)
sns.histplot(merged_data[['riv_human']],x = 'riv_human', color='orange', edgecolor='red', kde=True)
plt.title('Human scores')
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
sns.histplot(merged_data[['riv_original']],x = 'riv_original', kde=True)
plt.title('True scores')
plt.show()


