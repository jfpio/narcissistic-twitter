"""
# Human as a model

This is another way of baseline comparison where the psychologist is shown the same data as the model: post (travel) and two types of narcissism scores - the training dataset in total of 91 examples as the one below.

| post_travel         | adm     | riv |
|--------------|-----------|------------|
| I wish I could travel 24/7 and get paid for it | 1.444 | 1.111 |

And then has to assess the admiration and rivalry scores in the test dataset (44 examples) based on the posts. As shown below. 

| post_travel         | adm     | riv |
|--------------|-----------|------------|
| Roads were quiet on the way to London today. |  | |

Then the comparison between the true and predicted results is conducted. 

The final results show that human psychologists perform worse than Few-shot method.
"""

"""
## Load the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import torch
from torch.nn import HuberLoss


path_to_human_assessed_A = '../data/responses/human_as_model_travel_A.csv'
path_to_human_assessed_B = '../data/responses/human_as_model_travel_B.csv'
path_to_human_assessed_C = '../data/responses/human_as_model_travel_C.csv'

path_to_test = '../data/split/test.csv'


# load the human assessed data
human_data_A = pd.read_csv(path_to_human_assessed_A)
human_data_B = pd.read_csv(path_to_human_assessed_B)
human_data_C = pd.read_csv(path_to_human_assessed_C)
human_data_A.head()


# load the test data
test_data = pd.read_csv(path_to_test)
test_data[['post_travel','adm','riv']].head()


# drop not needed columns
test_data = test_data[['post_travel','adm','riv']]


# Merge the two dataframes
merged_data = test_data.merge(human_data_A, on='post_travel', suffixes=('', '_human_A')) \
                  .merge(human_data_B, on='post_travel', suffixes=('', '_human_B')) \
                  .merge(human_data_C, on='post_travel', suffixes=('', '_human_C'))
merged_data.rename(columns={'adm': 'adm_original', 'riv': 'riv_original'}, inplace=True)
# Check the length of the merged data
print(f"Merged correctly: {len(merged_data)==len(human_data_A)==len(human_data_B)==len(human_data_C)==len(test_data)}")
merged_data.head()


"""
# Mean squared error for the human assessed data
"""

# Calculate the mean squared error
rmse_adm_A = root_mean_squared_error(merged_data['adm_human_A'], merged_data['adm_original'])
rmse_riv_A = root_mean_squared_error(merged_data['riv_human_A'], merged_data['riv_original'])
rmse_adm_B = root_mean_squared_error(merged_data['adm_human_B'], merged_data['adm_original'])
rmse_riv_B = root_mean_squared_error(merged_data['riv_human_B'], merged_data['riv_original'])
rmse_adm_C = root_mean_squared_error(merged_data['adm_human_C'], merged_data['adm_original'])
rmse_riv_C = root_mean_squared_error(merged_data['riv_human_C'], merged_data['riv_original'])
print(f"Root mean Squared Error for adm: A:{rmse_adm_A}, B:{rmse_adm_B}, C:{rmse_adm_C},\n\
       mean: {np.mean([rmse_adm_A,rmse_adm_B,rmse_adm_C])}")
print(f"Root mean Squared Error for riv: A:{rmse_riv_A}, B:{rmse_riv_B}, C:{rmse_riv_C},\n\
       mean: {np.mean([rmse_riv_A,rmse_riv_B,rmse_riv_C])}")


"""
# Other metrics
"""

# other metrics
# Calculate the mean absolute error
mae_adm_A = mean_absolute_error(merged_data['adm_human_A'], merged_data['adm_original'])
mae_riv_A = mean_absolute_error(merged_data['riv_human_A'], merged_data['riv_original'])
mae_adm_B = mean_absolute_error(merged_data['adm_human_B'], merged_data['adm_original'])
mae_riv_B = mean_absolute_error(merged_data['riv_human_B'], merged_data['riv_original'])
mae_adm_C = mean_absolute_error(merged_data['adm_human_C'], merged_data['adm_original'])
mae_riv_C = mean_absolute_error(merged_data['riv_human_C'], merged_data['riv_original'])

print(f"Mean Absolute Error for adm: A:{mae_adm_A}, B:{mae_adm_B}, C:{mae_adm_C},\n\
       mean: {np.mean([mae_adm_A,mae_adm_B,mae_adm_C])}")
print(f"Mean Absolute Error for riv: A:{mae_riv_A}, B:{mae_riv_B}, C:{mae_riv_C},\n\
       mean: {np.mean([mae_riv_A,mae_riv_B,mae_riv_C])}")


# other metrics
# Calculate huber loss
huber_loss = HuberLoss()
huber_adm_A = huber_loss(torch.tensor(merged_data['adm_human_A']), torch.tensor(merged_data['adm_original']))
huber_adm_B = huber_loss(torch.tensor(merged_data['adm_human_B']), torch.tensor(merged_data['adm_original']))
huber_adm_C = huber_loss(torch.tensor(merged_data['adm_human_C']), torch.tensor(merged_data['adm_original']))
huber_riv_A = huber_loss(torch.tensor(merged_data['riv_human_A']), torch.tensor(merged_data['riv_original']))
huber_riv_B = huber_loss(torch.tensor(merged_data['riv_human_B']), torch.tensor(merged_data['riv_original']))
huber_riv_C = huber_loss(torch.tensor(merged_data['riv_human_C']), torch.tensor(merged_data['riv_original']))

print(f"Huber Loss for adm: A:{huber_adm_A}, B:{huber_adm_B}, C:{huber_adm_C}, \n\
       mean: {np.mean([huber_adm_A,huber_adm_B,huber_adm_C])}")
print(f"Huber Loss for riv: A:{huber_riv_A}, B:{huber_riv_B}, C:{huber_riv_C}, \n\
       mean: {np.mean([huber_riv_A,huber_riv_B,huber_riv_C])}")


def quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile=0.5):
    error = y_true - y_pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))


quan_adm_A = quantile_loss(torch.tensor(merged_data['adm_human_A']), torch.tensor(merged_data['adm_original']))
quan_adm_B = quantile_loss(torch.tensor(merged_data['adm_human_B']), torch.tensor(merged_data['adm_original']))
quan_adm_C = quantile_loss(torch.tensor(merged_data['adm_human_C']), torch.tensor(merged_data['adm_original']))
quan_riv_A = quantile_loss(torch.tensor(merged_data['riv_human_A']), torch.tensor(merged_data['riv_original']))
quan_riv_B = quantile_loss(torch.tensor(merged_data['riv_human_B']), torch.tensor(merged_data['riv_original']))
quan_riv_C = quantile_loss(torch.tensor(merged_data['riv_human_C']), torch.tensor(merged_data['riv_original']))

print(f"Quantile Loss for adm: A:{quan_adm_A}, B:{quan_adm_B}, C:{quan_adm_C}, \n\
       mean: {np.mean([quan_adm_A,quan_adm_B,quan_adm_C])}")
print(f"Quantile Loss for riv: A:{quan_riv_A}, B:{quan_riv_B}, C:{quan_riv_C}, \n\
       mean: {np.mean([quan_riv_A,quan_riv_B,quan_riv_C])}")


def maxae(y_true: np.ndarray, y_pred: np.ndarray):
    return np.max(np.abs(y_true - y_pred))


maxae_adm_A = maxae(merged_data['adm_human_A'], merged_data['adm_original'])
maxae_adm_B = maxae(merged_data['adm_human_B'], merged_data['adm_original'])
maxae_adm_C = maxae(merged_data['adm_human_C'], merged_data['adm_original'])
maxae_riv_A = maxae(merged_data['riv_human_A'], merged_data['riv_original'])
maxae_riv_B = maxae(merged_data['riv_human_B'], merged_data['riv_original'])
maxae_riv_C = maxae(merged_data['riv_human_C'], merged_data['riv_original'])

print(f"Max Absolute Error for adm: A:{maxae_adm_A}, B:{maxae_adm_B}, C:{maxae_adm_C},\n\
       mean: {np.mean([maxae_adm_A,maxae_adm_B,maxae_adm_C])}")
print(f"Max Absolute Error for riv: A:{maxae_riv_A}, B:{maxae_riv_B}, C:{maxae_riv_C},\n\
       mean: {np.mean([maxae_riv_A,maxae_riv_B,maxae_riv_C])}")


"""
### Distribution
"""

# Define variables for plotting
colors = ['orange', 'blue', 'green']
edgecolors = ['red', 'black', 'yellow']
labels = ['Human A', 'Human B', 'Human C']

# Plotting the histograms
plt.figure(figsize=(10, 6))

ax1 = plt.subplot(1, 2, 1)
for i, suffix in enumerate(['_human_A', '_human_B', '_human_C']):
    sns.histplot(merged_data[f'adm{suffix}'], color=colors[i], edgecolor=edgecolors[i], kde=True, label=labels[i])
plt.title('Human Scores')
plt.xlabel('ADM')
plt.legend()

# Plot for original data
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
sns.histplot(merged_data['adm_original'], kde=True, color='purple', label='Original')
plt.title('Original Scores')
plt.xlabel('ADM')
plt.legend()

plt.tight_layout()
plt.show()


# Define variables for plotting
colors = ['orange', 'blue', 'green']
edgecolors = ['red', 'black', 'yellow']
labels = ['Human A', 'Human B', 'Human C']

# Plotting the histograms
plt.figure(figsize=(10, 6))

ax1 = plt.subplot(1, 2, 1)
for i, suffix in enumerate(['_human_A', '_human_B', '_human_C']):
    sns.histplot(merged_data[f'riv{suffix}'], color=colors[i], edgecolor=edgecolors[i], kde=True, label=labels[i])
plt.title('Human Scores')
plt.xlabel('RIV')
plt.legend()

# Plot for original data
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
sns.histplot(merged_data['riv_original'], kde=True, color='purple', label='Original')
plt.title('Original Scores')
plt.xlabel('RIV')
plt.legend()

plt.tight_layout()
plt.show()


# Plotting histograms grouped together
plt.figure(figsize=(14, 8))

colors = ['orange', 'blue', 'green', 'purple']
labels = ['Scientist A', 'Scientist B', 'Scientist C', 'Original data']
columns = ['adm_human_A', 'adm_human_B', 'adm_human_C', 'adm_original']
columns_r = ['riv_human_A', 'riv_human_B', 'riv_human_C', 'riv_original']
xlabels = ['NARQ predictions','NARQ predictions','NARQ predictions','NARQ scores']
for i, column in enumerate(columns):
    plt.subplot(2, 2, i + 1)
    sns.histplot(merged_data[column], color=colors[i], kde=True)
    sns.histplot(merged_data[columns_r[i]], color='black', kde=True)

    plt.title(labels[i],fontsize=30)
    plt.xlabel(xlabels[i], fontsize=16)
    plt.ylabel('Count',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(1.0, 5.5, 0.5),fontsize=16)
    plt.legend(["Adm","Riv"])



plt.tight_layout()
plt.show()




