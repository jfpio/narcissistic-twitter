"""
# Statistical baseline
This notebook is calculating the statistical baselines. It takes mean as a predicted value and checks the results. The measures in questions are: RMSE, MAE, MaxAE, Huber Loss(q=1), Quantile loss(q=0.5). Both are calculated for a full test dataset. TL;DR: What is the score if we just give the mean as a prediction? 

Results for travel and abortion posts (they are the same)
|measure|adm|riv|
|-----|----|---|
|MAE|0.88|0.62|
|MaxAE|2.49|2.54|
|Huber Loss|0.49|0.27|
|Quantile Loss|0.44|0.31|
|RMSE|1.07|0.79|

Results for ai
|measure|adm|riv|
|-----|----|---|
|MAE|0.92|0.62|
|MaxAE|2.38|2.55|
|Huber Loss|0.53|0.28|
|Quantile Loss|0.46|0.31|
|RMSE|1.10| 0.81|
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

import torch
from torch.nn import HuberLoss


path_to_test = '../data/split/full_test.csv'


# load the test data
test_data = pd.read_csv(path_to_test)
test_data[['post_travel','adm','riv']].head()


# inictialize the huber loss
huber_loss = HuberLoss(delta=1.0)


# define the quantile loss
def quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile=0.5):
    error = y_true - y_pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))


"""
## Travel and abortion posts
### Narcissistic Admiration
"""

# get the mean of the adm column
mean_adm_tensor = np.full(test_data.shape[0], np.mean(test_data['adm']))


# calculate the mean absolute error
adm_mae = mean_absolute_error(test_data['adm'], mean_adm_tensor)
print(f'MAE: {adm_mae}')


adm_maxae = np.max(np.abs(test_data['adm'] - mean_adm_tensor))
print(f'MAXAE: {adm_maxae}')


adm_huber_loss_score = huber_loss(torch.tensor(test_data['adm'].values), 
                                  torch.tensor(mean_adm_tensor))

print(f'Huber Loss: {adm_huber_loss_score}')


adm_quantile_loss_score = quantile_loss(torch.tensor(test_data['adm'].values), 
                                        torch.tensor(mean_adm_tensor), quantile=0.5)

print(f'Quantile Loss: {adm_quantile_loss_score}')


# RMSE
adm_rmse = root_mean_squared_error(test_data['adm'], mean_adm_tensor)
print(f'RMSE: {adm_rmse}')


"""
### Narcissistic Rivalry
"""

# get the mean of the adm column
mean_riv_tensor = np.full(test_data.shape[0], np.mean(test_data['riv']))


riv_mae = mean_absolute_error(test_data['riv'], mean_riv_tensor)
print(f'MAE: {riv_mae}')


riv_maxae = np.max(np.abs(test_data['riv'] - mean_riv_tensor))
print(f'MAXAE: {riv_maxae}')


riv_huber_loss_score = huber_loss(torch.tensor(test_data['riv'].values), 
                                  torch.tensor(mean_riv_tensor))
print(f'Huber Loss: {riv_huber_loss_score}')


riv_quantile_loss_score = quantile_loss(torch.tensor(test_data['riv'].values),
                                        torch.tensor(mean_riv_tensor), quantile=0.5)
print(f'Quantile Loss: {riv_quantile_loss_score}')


riv_rmse = root_mean_squared_error(test_data['riv'], mean_riv_tensor)
print(f'RMSE: {riv_rmse}')


"""
## AI posts
### Admiration
"""

ai_test_data = test_data[['post_ai','adm','riv']].dropna()


mean_adm_tensor = np.full(ai_test_data.shape[0], np.mean(ai_test_data['adm']))


adm_mae = mean_absolute_error(ai_test_data['adm'], mean_adm_tensor)
print(f'MAE: {adm_mae}')


adm_maxae = np.max(np.abs(ai_test_data['adm'] - mean_adm_tensor))
print(f'MAXAE: {adm_maxae}')


adm_huber_loss_score = huber_loss(torch.tensor(ai_test_data['adm'].values),
                                    torch.tensor(mean_adm_tensor))
print(f'Huber Loss: {adm_huber_loss_score}')


adm_quantile_loss_score = quantile_loss(torch.tensor(ai_test_data['adm'].values),
                                        torch.tensor(mean_adm_tensor), quantile=0.5)
print(f'Quantile Loss: {adm_quantile_loss_score}')


adm_rmse = root_mean_squared_error(ai_test_data['adm'], mean_adm_tensor)
print(f'RMSE: {adm_rmse}')


"""
### Rivalry
"""

mean_riv_tensor = np.full(ai_test_data.shape[0], np.mean(ai_test_data['riv']))


riv_mae = mean_absolute_error(ai_test_data['riv'], mean_riv_tensor)
print(f'MAE: {riv_mae}')


riv_maxae = np.max(np.abs(ai_test_data['riv'] - mean_riv_tensor))
print(f'MAXAE: {riv_maxae}')


riv_huber_loss_score = huber_loss(torch.tensor(ai_test_data['riv'].values),
                                    torch.tensor(mean_riv_tensor))
print(f'Huber Loss: {riv_huber_loss_score}')


riv_quantile_loss_score = quantile_loss(torch.tensor(ai_test_data['riv'].values),
                                        torch.tensor(mean_riv_tensor), quantile=0.5)
print(f'Quantile Loss: {riv_quantile_loss_score}')


riv_rmse = root_mean_squared_error(ai_test_data['riv'], mean_riv_tensor)
print(f'RMSE: {riv_rmse}')




