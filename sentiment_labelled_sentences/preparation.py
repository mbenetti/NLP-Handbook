# NLP Sentiment Analysis Handbook
### A Step-By-Step Approach to Understand TextBlob, NLTK, Scikit-Learn, and LSTM 
### networks. See: 'https://towardsdatascience.com/nlp-sentiment-analysis-for-beginners-e7897f976897'

''' Introduction

Natural Language Processing (NLP) is the area of machine learning that focuses on the generation and understanding of language. Its main objective is to enable machines to understand, communicate and interact with humans in a natural way.

NLP has many tasks such as Text Generation, Text Classification, Machine Translation, Speech Recognition, Sentiment Analysis, etc. For a beginner to NLP, looking at these tasks and all the techniques involved in handling such tasks can be quite daunting. And in fact, it is very difficult for a newbie to know exactly where and how to start.

Out of all the NLP tasks, I personally think that Sentiment Analysis (SA) is probably the easiest, which makes it the most suitable starting point for anyone who wants to start go into NLP.

In this article, I compile various techniques of how to perform SA, ranging from simple ones like TextBlob and NLTK to more advanced ones like Sklearn and Long Short Term Memory (LSTM) networks.

After reading this, you can expect to understand the followings:

    Toolkits used in SA: TextBlob and NLTK
    Algorithms used in SA: Naive Bayes, SVM, Logistic Regression and LSTM
    Jargons like stop-word removal, stemming, bag of words, corpus, tokenisation etc.
    Create a word cloud

The flow of this article:

    Data cleaning and pre-processing
    TextBlob
    Algorithms: Logistic Regression, Naive Bayes, SVM and LSTM '''
    


#%%
%matplotlib inline
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

### with open("sentiment_labelled_sentences/full_set.txt") as f:
    content = f.readlines()content[0:10]
#%%
with open("full_set.txt") as f:
    content = f.readlines()
    
content[0:10]
# %%
## Remove leading and trailing white space
content = [x.strip() for x in content]

## Separate the sentences from the labels
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]
sentences[0:10]
labels[0:10]
# %%
