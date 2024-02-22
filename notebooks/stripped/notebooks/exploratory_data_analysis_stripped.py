import numpy as np
import pandas as pd

# Charts
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# NLP
import re
import nltk
from nltk import tokenize
from nltk.corpus import stopwords

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer


# downaload the stopwords
nltk.download('popular')


# Load the data
# TODO: Check the pyreadstat library to read directly from .sav file 
# Data path
data_path = 'data/teach AI 12.12.23.csv'
data = pd.read_csv(data_path, decimal=',')
data.head()


# Analyze the data
# Check the data types
data.dtypes


# List of not needed columns to drop:
columns_to_drop = ['ID','CNI_check','GCB_check_',
                   'Progress','Finished','Location']
data.drop(columns=columns_to_drop, inplace=True)
data.head()


# Change the values #NULL! to np.nan
data.replace('#NULL!', np.nan, inplace=True)
data.head(2)


# Check the missing values
data.isnull().sum()


# Check the number of unique values of each column
data.nunique()


# plot the box chart of ADM and RIV column
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['ADM', 'RIV']])
plt.title('Box plot of ADM and RIV')
plt.show()


# Plot gender distribution
plt.figure(figsize=(10, 6))
plt.pie(data['Gender'].value_counts(),labels=['Female','Male','Other']
        ,autopct='%.0f%%')
plt.show()


"""
## Text analisis
"""

data['Length_abor'] = data['POST___ABORTION'].str.len()
data['Length_trav'] = data['POST___TRAVEL'].str.len()
data[['POST___ABORTION','POST___TRAVEL','Length_abor','Length_trav']].head(5)


# word count function
def word_count(text):
    words = text.split()
    return len(words)


data['Word_count_abor'] = data['POST___ABORTION'].apply(word_count)
data['Word_count_trav'] = data['POST___TRAVEL'].apply(word_count)
data[['POST___ABORTION','POST___TRAVEL','Word_count_abor',
      'Word_count_trav']].head(5)


np.mean([len(sent) for sent in 
         tokenize.sent_tokenize(data['POST___ABORTION'][0])])


# mean sentence length lambda function
data['Mean_sentence_length_abor'] = data['POST___ABORTION'].apply(
    lambda x: np.mean([len(sent) for sent in tokenize.sent_tokenize(x)]))
data['Mean_sentence_length_trav'] = data['POST___TRAVEL'].apply(
    lambda x: np.mean([len(sent) for sent in tokenize.sent_tokenize(x)]))
data[['POST___ABORTION','POST___TRAVEL',
      'Mean_sentence_length_abor','Mean_sentence_length_trav']].head(5)


# Plot two distributions with legend
def visualize(col_ab, col_trav,title):
    plt.subplot(1,2,2)
    sns.kdeplot(data[col_ab], label='abortion')
    sns.kdeplot(data[col_trav], label='travel')
    legend = plt.legend()
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')


# visualize the distributions of the word count and the mean sentence length
visualize('Word_count_abor', 'Word_count_trav','Word count distribution')
plt.show()
visualize('Length_abor', 'Length_trav', 'Post length distribution')
plt.show()
visualize('Mean_sentence_length_abor', 
          'Mean_sentence_length_trav', 'Mean sentence length distribution')
plt.show()


# TODO: Find twitter statistics for comparison


"""
## Term Frequency Analysis
"""

def clean(review):
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() 
                       if word not in stopwords.words('english')])
    return review


data['POST___ABORTION'] = data['POST___ABORTION'].apply(clean)
data['POST___TRAVEL'] = data['POST___TRAVEL'].apply(clean)
data.head(2)


def corpus(text):
    text_list = text.split()
    return text_list


data['POST___ABORTION_lists'] = data['POST___ABORTION'].apply(corpus)
data['POST___TRAVEL_lists'] = data['POST___TRAVEL'].apply(corpus)
data.head(2)


corpus_ab = []
corpus_tr = []
for text in data['POST___ABORTION_lists']:
    corpus_ab += text
for text in data['POST___TRAVEL_lists']:
    corpus_tr += text
print(f"Abortion total words:\t{len(corpus_ab)} \nTravel total words:\t{len(corpus_tr)}")


mostCommon_ab = Counter(corpus_ab).most_common(10)
mostCommon_tr = Counter(corpus_tr).most_common(10)

words = []
freq = []
for word, count in mostCommon_tr:
    words.append(word)
    freq.append(count)

sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occuring Words in Travel Posts')
plt.show()


words = []
freq = []
for word, count in mostCommon_ab:
    words.append(word)
    freq.append(count)

sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occuring Words in Travel Posts')
plt.show()


def bigram_freq(column):
    cv = CountVectorizer(ngram_range=(2,2))
    bigrams = cv.fit_transform(column)
    count_values = bigrams.toarray().sum(axis=0)
    bigram_freq = pd.DataFrame(
        sorted([(count_values[i], k) 
                for k, i in cv.vocabulary_.items()], reverse = True))
    bigram_freq.columns = ["frequency", "bigram"]
    return bigram_freq


cv = CountVectorizer(ngram_range=(2,2))
bigrams_ab_freq = bigram_freq(data['POST___ABORTION'])
bigrams_tr_freq = bigram_freq(data['POST___TRAVEL'])

sns.barplot(x=bigrams_tr_freq['frequency'][:10], 
            y=bigrams_tr_freq['bigram'][:10])
plt.title('Top 10 Most Frequently Occuring Bigrams Travel Posts')
plt.show()

sns.barplot(x=bigrams_ab_freq['frequency'][:10], 
            y=bigrams_ab_freq['bigram'][:10])
plt.title('Top 10 Most Frequently Occuring Bigrams Abbortion Posts')
plt.show()


columns_for_corr_1 = ['ADM', 'RIV', 'Length_abor', 'Length_trav', 'Word_count_abor', 'Word_count_trav', 'Mean_sentence_length_abor', 'Mean_sentence_length_trav']
columns_for_corr_2 = ['ADM', 'RIV','Gender','Age', 'Education', 'Marital_status','Employment','Twitter']


sns.heatmap(data[columns_for_corr_1].corr(), annot=True)
plt.show()


sns.heatmap(data[columns_for_corr_2].corr(), annot=True)
plt.show()




