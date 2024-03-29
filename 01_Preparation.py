#!/usr/bin/env python
# coding: utf-8

# <p align="center">
#   <img width="900" height="500" src="figures/01.jpg">
# </p>
# 
# # NLP Sentiment Analysis Handbook <!-- omit in toc -->
# 
# A Step-By-Step Approach to Understand TextBlob, NLTK, Scikit-Learn, and LSTM  networks 
# applied to Sentiment Analysis.
# 
# 
# Adaptation, corrections, and modifications by Mauro Benetti 03-2021.
# 
# This repository is based on :
# 
# * https://towardsdatascience.com/nlp-sentiment-analysis-for-beginners-e7897f976897
# * https://www.mdeditor.tw/pl/pISR/zh-hk
# * https://stackabuse.com/removing-stop-words-from-strings-in-python/
# 
# For additional information please visit:
# * https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
# * http://www.nltk.org/data.html
# 
# -------------------------------------------------------------------------------------
# 
# # Introduction
# 
# **Natural language processing** (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural-language generation [(Wikipedia,2020)](https://en.wikipedia.org/wiki/Natural_language_processing). 
# 
# NLP has many tasks such as Text Generation, Text Classification, Machine Translation, Speech Recognition, Sentiment Analysis, etc. For a beginner to NLP, looking at these tasks and all the techniques involved in handling such tasks can be quite daunting. 
# And in fact, it is very difficult for a newbie to know exactly where and how to start.
# 
# Out of all the NLP tasks, I personally think that Sentiment Analysis (SA) is probably the easiest, which makes it the most suitable starting point for anyone who wants to start go into NLP.
# 
# In this repository, I compile various techniques of how to perform SA, ranging from simple ones like TextBlob and NLTK to more advanced ones like Sklearn and Long Short Term Memory (LSTM) networks.
# 
# After reading this, you can expect to understand the followings:
# 
# *   Toolkits used in SA: TextBlob and NLTK
# *   Algorithms used in SA: Naive Bayes, SVM, Logistic Regression and LSTM
# *   Jargons like stop-word removal, stemming, bag of words, corpus, tokenization etc.
# *   Create a word cloud.
# 
# The flow of this article:
# 
# *   Data cleaning and pre-processing
# *   TextBlob
# *   Algorithms: Logistic Regression, Naive Bayes, SVM and LSTM
#     
# ## Problem Formulation
# In this repository, we will work with a data set that consists of 3000 sentences coming from reviews on imdb.com, amazon.com, and yelp.com. Each sentence is labeled according to whether it comes from a positive review (labelled as 1) or negative review (labelled as 0).The folder sentiment_labelled_sentences(containing the data file full_set.txt) should be in the same directory as your notebook/script.
# 
# 

# # Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)


# # Loading the data

# The file **full_set.txt** should be in the same directory as your notebook.

# In[2]:


with open("sentiment_labelled_sentences/full_set.txt") as f:
    content = f.readlines()
    
content[0:10]


# # Pre-processing

# ## Removing leading and trailing whitespaces

# In[3]:


content = [x.strip() for x in content]


# ## Separate the sentences from the labels

# In[4]:


sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]

sentences[0:10]


# In[5]:


labels[0:10]


# ## Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
#  
# Where (-1) represents negative and (1) represents positive 
# 

# In[6]:


y = np.array(labels, dtype='int8')
y = 2*y - 1


# To input data into the any model, the data input must be in vector form. We will do the 
# following transformations:
# 
# * Remove punctuation and numbers
# * Transform all words to lower-case
# * Remove stop words (e.g. the, a, that, this, it, …)
# * Tokenizer the texts
# * Convert the sentences into vectors, using a bag-of-words representation
# 

# In[7]:


def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

stoppers = ['a', 'is', 'of','the','this','uhm','uh']

removeStopWords(stoppers, "this is a test of the stop word removal code")


# ## Using NLTK

# In case you don't have the list of stopwords for english language

# In[8]:

#Only run this line the first time, run python command and then on the >>>
# import nltk

# nltk.download(stopwords) 


# In[9]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


# Removing the stop words with NLTK

# In[10]:


removeStopWords(stops, "this is a test of the stop word removal code.")


# In[11]:


word_list = sentences
filtered_words = [word for word in word_list if word not in stops]


# ## All together

# In[12]:
# Using the stopwords.
from nltk.corpus import stopwords

# Initialize the stopwords
stoplist = stopwords.words('english')

def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x

### Remove digits 
digits = [str(x) for x in range(10)]
remove_digits = [full_remove(x, digits) for x in sentences]

# ### Remove punctuation 
remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]

# ### Make everything lower-case and remove any white space ##
sents_lower = [x.lower() for x in remove_punc]
sents_lower = [x.strip() for x in sents_lower]

# ### Remove stop words 
from nltk.corpus import stopwords
stops = stopwords.words("english")

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

sents_processed = [removeStopWords(stops,x) for x in sents_lower]

# ### Results
sents_processed[0:20]


# ## How to remove a list of specific stops from the sentences 
# 
# You can add or remove stop words as per your choice to the existing collection of stop 
# words in NLTK. Before removing or adding stop words in NLTK, let's see the list of all 
# the English stop words supported by NLTK:

# In[13]:


print(stopwords.words('english'))


# In[14]:


stop_set = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
sents_processed = [removeStopWords(stop_set,x) for x in sents_lower]
sents_processed[0:20]


# ### How to adding Stop Words to Default NLTK Stop Word List
# 
# To add a word to NLTK stop words collection, first create an object from the stopwords.
# words('english') list. Next, use the append() method on the list to add any word to the 
# list.
# 
# The following script adds the word play to the NLTK stop word collection. Again, we 
# remove all the words from our text variable to see if the word play is removed or not.

# In[15]:


all_stopwords = stopwords.words('english')
all_stopwords.append('play')
print(all_stopwords)


# ### Removing Stop Words from the default NLTK Stop Word List

# In[16]:


all_stopwords.remove('not')
print(all_stopwords)


# ## Tokenization and steamming
# 
# It is ok to stop here and move to Tokenization. However, one can continue with stemming. 
# The goal of stemming is too strip off prefixes and suffixes in the word and convert the 
# word into its base form, e.g. studying->study, beautiful->beauty, cared->care, …In NLTK, 
# there are 2 popular stemming techniques called porter and lanscaster.
# 
# (https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)
# 

# In[17]:


import nltk
def stem_with_porter(words):
    porter = nltk.PorterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words
    
def stem_with_lancaster(words):
    porter = nltk.LancasterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words    ## Demonstrate ##    
str = "Please don't unbuckle your seat-belt while I am driving, he said"

print("porter:", stem_with_porter(str.split()))

print("lancaster:", stem_with_lancaster(str.split()))


# 
# **Let’s try on our sents_processed to see whether it makes sense**

# In[20]:


porter = [stem_with_porter(x.split()) for x in sents_processed]

porter = [" ".join(i) for i in porter]

porter[0:10]


# In[21]:


lancaster = [stem_with_lancaster(x.split()) for x in sents_processed]

lancaster = [" ".join(i) for i in lancaster]

lancaster[0:10]


# **Some weird changes occur, e.g. very->veri, quality->qualiti, value->valu, …**

# # TD/IDF
# 
# Term Document Inverse Document Frequency (TD/IDF). This is a measure of the relative 
# importance of a word within a document, in the context of multiple documents . In our 
# case here, multiple reviews.
# 
# We start with the TD part — this is simply a normalized frequency of the word in the 
# document:
# 
# (word count in document) / (total words in document)
# The IDF is a weighting of the uniquess of the word across all of the documents. Here is 
# the complete formula of TD/IDF:
# 
#                      td_idf(t,d) = wc(t,d)/wc(d) / dc(t)/dc()
# 
# where:
# 
# * wc(t,d) = # of occurrences of term t in doc d
# 
# * wc(d) = # of words in doc d
# 
# * dc(t) = # of docs that contain at least 1 occurrence of term t
# 
# * dc() = # of docs in collection

# Now, let’s create a bag of words and normalise the texts

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer(analyzer = "word", preprocessor = None, stop_words =  'english', max_features = 6000, ngram_range=(1,5))

data_features = vectorizer.fit_transform(sents_processed)
tfidf_transformer = TfidfTransformer()
data_features_tfidf = tfidf_transformer.fit_transform(data_features)
data_mat = data_features_tfidf.toarray()


# Now data_mat is our Document-Term matrix. Input is ready to put into model. Let’s create 
# Training and Test sets. Here, I split the data into a training set of 2500 sentences and 
# a test set of 500 sentences (of which 250 are positive and 250 negative). 

# ## Train and test dataset

# In[24]:


np.random.seed(0)
test_index = np.append(np.random.choice((np.where(y==-1))[0], 250, replace=False), np.random.choice((np.where(y==1))[0], 250, replace=False))
train_index = list(set(range(len(labels))) - set(test_index))
train_data = data_mat[train_index,]
train_labels = y[train_index]
test_data = data_mat[test_index,]
test_labels = y[test_index]


# -------------------------------------------------------------------------------------
# # TextBlob
# 
# TextBlob : Linguistic researchers have labeled the sentiment of words based on their 
# domain expertise. Sentiment of words can vary based on where it is in a sentence. The 
# TextBlob module allows us to take advantage of these labels. TextBlod finds all the words 
# and phrases that it can assign polarity and subjectivity to, and average all of them 
# together.
# 
# Sentiment Labels : Each word in a corpus is labeled in terms of polarity and subjectivity 
# (there are more labels as well, but we’re going to ignore them for now). A corpus’ 
# sentiment is the average of these.
# 
# Polarity : How positive or negative a word is. -1 is very negative. +1 is very positive.
# Subjectivity : How subjective, or opinionated a word is. 0 is fact. +1 is very much 
# an opinion.
# 

# 
# ## Create polarity function and subjectivity function

# In[28]:


from textblob import TextBlob
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
pol_list = [pol(x) for x in sents_processed]
sub_list = [sub(x) for x in sents_processed]


# In[29]:



#-------------------------------------------------------------------------------------


# In[30]:



#-------------------------------------------------------------------------------------


# In[31]:



#-------------------------------------------------------------------------------------


# In[32]:



#-------------------------------------------------------------------------------------


# In[33]:



#-------------------------------------------------------------------------------------


