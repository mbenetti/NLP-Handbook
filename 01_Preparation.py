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
    
''' Problem Formulation

In this article, I will the sentiment data set that consists of 3000 sentences coming from reviews on imdb.com, amazon.com, and yelp.com. Each sentence is labeled according to whether it comes from a positive review (labelled as 1) or negative review (labelled as 0).

Data can be downloaded from the website. Alternatively, it can be downloaded from here (highly recommended). The folder sentiment_labelled_sentences(containing the data file full_set.txt) should be in the same directory as your notebook.
 '''

#### Libraries
#%%
%matplotlib inline
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

#### Loading data
#%%
with open("sentiment_labelled_sentences/full_set.txt") as f:
    content = f.readlines()
    
content[0:10]

#-------------------------------------------------------------------------------------


# %%
#### Pre-processing

### Remove leading and trailing white space
content = [x.strip() for x in content]

### Separate the sentences from the labels
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]


#%%
sentences[0:10]


#%%
labels[0:10]

### Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
''' One can stop here for this section. But for me, I prefer transforming y into (-1,1) 
form, where -1 represents negative and 1 represents positive '''


#%%

y = np.array(labels, dtype='int8')
y = 2*y - 1

''' 
To input data into the any model, the data input must be in vector form. We will do the following transformations:

* Remove punctuation and numbers
* Transform all words to lower-case
* Remove stop words (e.g. the, a, that, this, it, …)
* Tokenizer the texts
* Convert the sentences into vectors, using a bag-of-words representation

#### Definitions

* Stop words *: common words that are not interesting for the task at hand. These usually include articles such as ‘a’ and ‘the’, pronouns such as ‘i’ and ‘they’, and prepositions such as ‘to’ and ‘from’, …
 '''
 
#-------------------------------------------------------------------------------------


# %%
### Removing the stop words by defining a list of sentences, this case a sentence is one element in the list

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

stoppers = ['a', 'is', 'of','the','this','uhm','uh']

removeStopWords(stoppers, "this is a test of the stop word removal code")

#-------------------------------------------------------------------------------------


# %%
### With NLTK, first run this two commands on a python terminal

# import nltk
# nltk.download(stopwords)
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

#-------------------------------------------------------------------------------------


#%%
### Removing the stop words with NLTK

removeStopWords(stops, "this is a test of the stop word removal code.")

#%%
### In case a have a list with words already

word_list = sentences
filtered_words = [word for word in word_list if word not in stops]


# %%
### All together

def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x

### Remove digits ##
digits = [str(x) for x in range(10)]
remove_digits = [full_remove(x, digits) for x in sentences]

### Remove punctuation ##
remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]

### Make everything lower-case and remove any white space ##
sents_lower = [x.lower() for x in remove_punc]
sents_lower = [x.strip() for x in sents_lower]

### Remove stop words ##
from nltk.corpus import stopwords
stops = stopwords.words("English")

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

sents_processed = [removeStopWords(stops,x) for x in sents_lower]

## Results
sents_processed[0:20]
#-------------------------------------------------------------------------------------


# %%
# To remove a list of specific stops from the sentences 

stop_set = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
sents_processed = [removeStopWords(stop_set,x) for x in sents_lower]
sents_processed[0:20]
#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%
#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------


# %%

#-------------------------------------------------------------------------------------

