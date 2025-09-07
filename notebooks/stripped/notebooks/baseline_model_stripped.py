"""
# Bag of words baseline model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from matplotlib import pyplot as plt


post_type = 'post_abortion'
data_path = '../data/split/full_train.csv'
test_data_path = '../data/split/full_test.csv'
narcism_type = 'adm'


# Read the data
data = pd.read_csv(data_path)
test_data = pd.read_csv(test_data_path)
data.head()


# Create a pipeline that includes vectorization and transformation
preprocessing_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

tfidf = preprocessing_pipeline.fit_transform(data[post_type].values)
df_counts = pd.DataFrame(tfidf.toarray())
df_counts['narcissism'] = data[narcism_type]

df_counts.head()


# Transform the test data
tfidf = preprocessing_pipeline.transform(test_data[post_type].values)
test_counts = pd.DataFrame(tfidf.toarray())
test_counts['narcissism'] = test_data[narcism_type]

test_counts.head()


"""
## Compare different models
"""

# Load split data
X_train = df_counts.iloc[:, :-1]
y_train = np.ravel(df_counts[['narcissism']])
X_test = test_counts.iloc[:, :-1]
y_test = np.ravel(test_counts[['narcissism']])


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def evaluate(y_test, y_pred):
    return root_mean_squared_error(y_test, y_pred)

def huber_loss(y_true, y_pred, delta=1.0):
    residual = y_true - y_pred
    is_small_error = np.abs(residual) <= delta
    squared_loss = 0.5 * residual**2
    linear_loss = delta * (np.abs(residual) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

def quantile_loss(y_true, y_pred, quantile=0.5):
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))

def maxAE(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

def meanAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))



# Statistical baseline

def statistical_baseline(y_train, y_test):
    mean_value = np.mean(y_train)
    y_pred = np.full_like(y_test, mean_value)
    rmse = evaluate(y_test, y_pred)
    return rmse, y_pred

rmse, y_pred = statistical_baseline(y_train, y_test)

print(f'Statistical baseline RMSE: {rmse:.4f}')
print(f'huber_loss: {huber_loss(y_test, y_pred):.4f}')
print(f'quantile_loss: {quantile_loss(y_test, y_pred):.4f}')
print(f'maxAE: {maxAE(y_test, y_pred):.4f}')
print(f'meanAE: {meanAE(y_test, y_pred):.4f}')



lr_model = LinearRegression()
train_model(lr_model, X_train, y_train)
y_pred = predict(lr_model, X_test)
rmse = evaluate(y_test, y_pred)
print(f"Linear Regression MSE: {rmse}")
print(f"huber loss: {huber_loss(y_test, y_pred)}")
print(f"quantile loss: {quantile_loss(y_test, y_pred)}")
print(f"maxAE: {maxAE(y_test, y_pred)}")
print(f"meanAE: {meanAE(y_test, y_pred)}")


mlpr_model = MLPRegressor(max_iter=800)
train_model(mlpr_model, X_train, y_train)
y_pred = predict(mlpr_model, X_test)
rmse = evaluate(y_test, y_pred)
print(f"MLP Regressor MSE: {rmse}")
print(f"huber loss: {huber_loss(y_test, y_pred)}")
print(f"quantile loss: {quantile_loss(y_test, y_pred)}")
print(f"maxAE: {maxAE(y_test, y_pred)}")
print(f"meanAE: {meanAE(y_test, y_pred)}")


svr_model = SVR()
train_model(svr_model, X_train, y_train)
y_pred = predict(svr_model, X_test)
rmse = evaluate(y_test, y_pred)
print(f"SVR MSE: {rmse}")
print(f"huber loss: {huber_loss(y_test, y_pred)}")
print(f"quantile loss: {quantile_loss(y_test, y_pred)}")
print(f"maxAE: {maxAE(y_test, y_pred)}")
print(f"meanAE: {meanAE(y_test, y_pred)}")


rfr_model = RandomForestRegressor()
train_model(rfr_model, X_train, y_train)
y_pred = predict(rfr_model, X_test)
rmse = evaluate(y_test, y_pred)
print(f"Random Forest MSE: {rmse}")
print(f"huber loss: {huber_loss(y_test, y_pred)}")
print(f"quantile loss: {quantile_loss(y_test, y_pred)}")
print(f"maxAE: {maxAE(y_test, y_pred)}")
print(f"meanAE: {meanAE(y_test, y_pred)}")


dtr_model = DecisionTreeRegressor()
train_model(dtr_model, X_train, y_train)
y_pred = predict(dtr_model, X_test)
rmse = evaluate(y_test, y_pred)
print(f"Decision Tree MSE: {rmse}")
print(f"huber loss: {huber_loss(y_test, y_pred)}")
print(f"quantile loss: {quantile_loss(y_test, y_pred)}")
print(f"maxAE: {maxAE(y_test, y_pred)}")
print(f"meanAE: {meanAE(y_test, y_pred)}")


"""
So far the best effects give SVM and Random Forest, but also the simples solution has rather small MSE - LinearRegression
"""

"""
## Testing on some new data
Generated by chat GPT 3.5 
Prompt: *"Write me a travel post for twitter"*
"""

# Choosing the best model based on the MSE
best_model = svr_model


# Won't work with the current data

new_data = ["Embarking on an exhilarating adventure through the enchanting streets of Kyoto, Japan. 🎌 From the serene bamboo forests of Arashiyama to the historic temples of Kinkaku-ji and Fushimi Inari Taisha, every corner unveils a tale of tradition and tranquility. #Kyoto #TravelJapan", "Lost in the colorful labyrinth of Marrakech's bustling souks, where the scent of spices fills the air and vibrant textiles dance in the breeze. 🕌✨ Exploring hidden riads, savoring tagine delights, and getting lost in the magic of Jardin Majorelle. #Marrakech #TravelGoals 🌴🌞"]
preprocessed_data = preprocessing_pipeline.transform(new_data)
post_test = pd.DataFrame(preprocessed_data.toarray())
predicted = predict(best_model, post_test)
for doc, category in zip(new_data, predicted):
     print(f'{doc} =>\nnarcism: {category}')


"""
Testing it on data that is not talking about travel (the abortion posts)
"""

tfidf = preprocessing_pipeline.transform(data["post_abortion"].values)

test_counts_ab = pd.DataFrame(tfidf.toarray())
test_counts_ab['narcissism'] = data[narcism_type]
X_test_ab = test_counts_ab.iloc[:, :-1]
y_test_ab = np.ravel(test_counts_ab[['narcissism']])
y_pred_ab = predict(best_model, X_test_ab)
rmse = evaluate(y_test_ab, y_pred_ab)
print(f"SVR RMSE test_ab: {rmse}")


"""
It is interesting that the model rather correctly predicts the 'ADM' narcissism in abortion posts despite being trained on different data. MSE is still less then 1 point (on the scale 1 to 6)
"""

"""
# Baseline model results
"""

def plot_mse_scatter(models, val_mse, test_mse, other_model,title, colors, labels, riv):
    # Combine the data into a list of tuples for sorting
    data = list(zip(models, val_mse, test_mse,other_model))
    
    # Sort the data by test MSE
    sorted_data = sorted(data, key=lambda x: x[2])
    
    # Unpack the sorted data back into separate lists
    sorted_models, sorted_val_mse, sorted_test_mse, other_model_mse = zip(*sorted_data)
    
    # Convert the sorted models list to a numeric scale for plotting
    x = range(len(sorted_models))
    
    # Create a scatter plot with smaller figure size and larger fonts
    plt.figure(figsize=(8, 6))
    
    # Plot validation MSE
    plt.scatter(x, sorted_val_mse, color=colors[0], label=labels[0], s=100, marker='o', ) # edgecolor='k'
    
    # Plot test MSE
    plt.scatter(x, sorted_test_mse, color=colors[1], label=labels[1], s=100, marker='s') # edgecolor='k'
    
    # Plot test MSE
    plt.scatter(x, other_model_mse, color=colors[2], label=labels[2], s=100, marker='D') # edgecolor='k'

    if riv:
        # Add horizontal lines for the Huber loss values
        plt.axhline(y=0.27, color='gray', linestyle='--', label='statistical baseline')
    else:
        # Add horizontal lines for the Huber loss values
        plt.axhline(y=0.49, color='gray', linestyle='--', label='statistical baseline')

    # Add labels and title with larger font size
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Huber loss', fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(x, sorted_models, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12, loc ='upper left')

    plt.ylim(0.25, 0.8)
    
    # Adding grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True)  # Move grid lines to the background
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Data taken from Nepute
# It is no longer mse, it is huber loss
# abortion - red, ai - blue, travel - green
models = ["LR", "SVR", "RFR", "GBR", "MLPR", "DTR"]
test_mse_adm_a = [0.517, 0.434, 0.444, 0.462, 0.496, 0.736]
test_mse_adm_a_2 = [0.629, 0.502, 0.511, 0.524, 0.640, 0.762]
test_mse_adm_a_3 = [0.593, 0.544, 0.592, 0.619, 0.623, 0.735]

test_mse_riv_a = [0.359, 0.274, 0.293, 0.342, 0.349, 0.488]
test_mse_riv_a_2 = [0.394, 0.283, 0.294, 0.329, 0.368, 0.415]
test_mse_riv_a_3 = [0.395, 0.290, 0.293, 0.342, 0.388, 0.470]

# This first
test_mse_riv_t = [0.428, 0.268, 0.291, 0.321, 0.353, 0.384]
test_mse_riv_t_2 = [0.519, 0.294, 0.301, 0.332, 0.433, 0.447]
test_mse_riv_t_3 = [0.534, 0.278, 0.276, 0.292, 0.355, 0.357]

test_mse_adm_t = [0.599, 0.487, 0.517, 0.559, 0.598, 0.767]
test_mse_adm_t_2 = [0.529, 0.492, 0.511, 0.549, 0.538, 0.714]
test_mse_adm_t_3 = [0.553, 0.549, 0.608, 0.576, 0.575, 0.770]

test_hub_riv_ai = [0.361, 0.283, 0.296, 0.341, 0.354, 0.480]
test_hub_riv_ai_2 = [0.322, 0.266, 0.284, 0.307, 0.313, 0.342]
test_hub_riv_ai_3 = [0.346, 0.274, 0.284, 0.292, 0.334, 0.441]

test_hub_adm_ai = [0.542, 0.451, 0.468, 0.516, 0.531, 0.686]
test_hub_adm_ai_2 = [0.559, 0.494, 0.505, 0.560, 0.557, 0.997]
test_hub_adm_ai_3 = [0.493, 0.463, 0.468, 0.477, 0.500, 0.756]


colors = ['#51EA30', '#3D36EB', '#EB7431'] #green-ai, blue-travel, orange-abortion

markers = ['s', 'D', 'o']  # square - AI, diamond - abortion, circle - travel

labels_tr = ["test (travel)", "test second category (abortion)", "test third category (ai)"]
labels_ab = ["test (abortion)", "test second category (travel)", "test third category (ai)"]
labels_ai = ["test (ai)", "test second category (travel)", "test third category (abortion)"]


titles = ["Rivarly prediction trained on abortion posts", "Rivarly prediction trained on travel posts",
            "Admiration prediction trained on abortion posts", "Admiration prediction trained on travel posts",
            "Rivarly prediction trained on AI posts", "Admiration prediction trained on AI posts"]


# Calling the function
plot_mse_scatter(models, test_mse_riv_a, test_mse_riv_a_2, test_mse_riv_a_3,titles[0], 
                 [colors[2], colors[1], colors[0]], labels_ab,True)
plot_mse_scatter(models, test_mse_riv_t, test_mse_riv_t_2, test_mse_riv_t_3,titles[1],
                 [colors[1], colors[2], colors[0]], labels_tr, True)
plot_mse_scatter(models, test_mse_adm_a, test_mse_adm_a_2, test_mse_adm_a_3,titles[2]
                 ,[colors[2], colors[1], colors[0]], labels_ab, False)

plot_mse_scatter(models, test_mse_adm_t, test_mse_adm_t_2, test_mse_adm_t_3,titles[3],
                 [colors[1], colors[2], colors[0]], labels_tr, False)
plot_mse_scatter(models, test_hub_riv_ai, test_hub_riv_ai_2, test_hub_riv_ai_3,titles[4],
                 [colors[0], colors[1], colors[2]], labels_ai, True)
plot_mse_scatter(models, test_hub_adm_ai, test_hub_adm_ai_2, test_hub_adm_ai_3,titles[5],
                 [colors[0], colors[1], colors[2]], labels_ai, False)




