#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES/DATASETS AND PERFORM EXPLORATORY DATA ANALYSIS

# In[ ]:


# import key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import plotly.express as px

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# In[4]:


get_ipython().system('pip install nltk')


# In[ ]:


# Mount the google drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# install nltk
# NLTK: Natural Language tool kit
get_ipython().system('pip install nltk')


# In[ ]:


# install gensim
# Gensim is an open-source library for unsupervised topic modeling and natural language processing
# Gensim is implemented in Python and Cython.
get_ipython().system('pip install gensim')


# In[ ]:


# load the stock news data
stock_df = pd.read_csv("C:\Finance\stock_sentiment.csv")


# In[ ]:


# Let's view the dataset 
stock_df


# In[ ]:


# dataframe information
stock_df.info()


# In[ ]:


# check for null values
stock_df.isnull().sum()


# In[ ]:





# # PERFORM DATA CLEANING (REMOVE PUNCTUATIONS FROM TEXT)
# 
# 

# In[ ]:


import string
string.punctuation


# In[ ]:


Test = '$I love AI & Machine learning!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[ ]:


Test = 'Good morning beautiful people :)... #I am having fun learning Finance with Python!!'


# In[ ]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[ ]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# In[ ]:


# Let's define a function to remove punctuations
def remove_punc(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)

    return Test_punc_removed_join


# In[ ]:


# Let's remove punctuations from our dataset 
stock_df['Text Without Punctuation'] = stock_df['Text'].apply(remove_punc)


# In[ ]:


stock_df


# In[ ]:


stock_df['Text'][2]


# In[ ]:


stock_df['Text Without Punctuation'][2]


# In[ ]:





# In[ ]:





# #  PERFORM DATA CLEANING (REMOVE STOPWORDS)

# In[ ]:


# download stopwords
nltk.download("stopwords")
stopwords.words('english')


# In[ ]:


# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])
# stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year', 'https'])


# In[ ]:


# Remove stopwords and remove short words (less than 2 characters)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= 3 and token not in stop_words:
            result.append(token)
            
    return result


# In[ ]:


# apply pre-processing to the text column
stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)


# In[ ]:


stock_df['Text'][0]


# In[ ]:


stock_df['Text Without Punc & Stopwords'][0]


# In[ ]:


# join the words into a string
# stock_df['Processed Text 2'] = stock_df['Processed Text 2'].apply(lambda x: " ".join(x))


# In[ ]:


stock_df


# In[ ]:





# #  PLOT WORDCLOUD

# In[ ]:


# join the words into a string
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text Without Punc & Stopwords'].apply(lambda x: " ".join(x))


# In[ ]:


# plot the word cloud for text with positive sentiment
plt.figure(figsize = (20, 20)) 
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(stock_df[stock_df['Sentiment'] == 1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation = 'bilinear');


# #  VISUALIZE CLEANED DATASETS

# In[ ]:


stock_df


# In[ ]:


nltk.download('punkt')


# In[ ]:


# word_tokenize is used to break up a string into words
print(stock_df['Text Without Punc & Stopwords Joined'][0])
print(nltk.word_tokenize(stock_df['Text Without Punc & Stopwords Joined'][0]))


# In[ ]:


# Obtain the maximum length of data in the document
# This will be later used when word embeddings are generated
maxlen = -1
for doc in stock_df['Text Without Punc & Stopwords Joined']:
    tokens = nltk.word_tokenize(doc)
    if(maxlen < len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is:", maxlen)


# In[ ]:


tweets_length = [ len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined'] ]
tweets_length


# In[ ]:


# Plot the distribution for the number of words in a text
fig = px.histogram(x = tweets_length, nbins = 50)
fig.show()


# In[ ]:





# #  PREPARE THE DATA BY TOKENIZING AND PADDING

# In[ ]:


stock_df


# In[ ]:


# Obtain the total words present in the dataset
list_of_words = []
for i in stock_df['Text Without Punc & Stopwords']:
    for j in i:
        list_of_words.append(j)


# In[ ]:


list_of_words


# In[ ]:


# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))
total_words


# In[ ]:


# split the data into test and train 
X = stock_df['Text Without Punc & Stopwords']
y = stock_df['Sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train


# In[ ]:


# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(X_train)

# Training data
train_sequences = tokenizer.texts_to_sequences(X_train)

# Testing data
test_sequences = tokenizer.texts_to_sequences(X_test)


# In[ ]:


train_sequences


# In[ ]:


test_sequences


# In[ ]:


print("The encoding for document\n", X_train[1:2],"\n is: ", train_sequences[1])


# In[ ]:


# Add padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 29, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 29, truncating = 'post')


# In[ ]:


for i, doc in enumerate(padded_train[:3]):
     print("The padded encoding for document:", i+1," is:", doc)


# In[ ]:


# Convert the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[ ]:


y_train_cat.shape


# In[ ]:


y_test_cat.shape


# In[ ]:


y_train_cat


# In[ ]:


# Add padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 15, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 15, truncating = 'post')


# #  BUILD A CUSTOM-BASED DEEP NEURAL NETWORK TO PERFORM SENTIMENT ANALYSIS

# In[ ]:


# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim = 512))

# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()


# In[ ]:


# train the model
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 2)


# In[ ]:





# # ASSESS TRAINED MODEL PERFORMANCE

# In[ ]:


# make prediction
pred = model.predict(padded_test)


# In[ ]:


# make prediction
prediction = []
for i in pred:
  prediction.append(np.argmax(i))


# In[ ]:


# list containing original values
original = []
for i in y_test_cat:
  original.append(np.argmax(i))


# In[ ]:


# acuracy score on text data
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(original, prediction)
accuracy


# In[ ]:


# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot = True)


# In[ ]:




