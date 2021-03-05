![Credit Pixabay](https://cdn.pixabay.com/photo/2016/04/30/13/12/sutterlin-1362879_1280.jpg)

# NLP Sentiment Analysis Handbook

A Step-By-Step Approach to Understand TextBlob, NLTK, Scikit-Learn, and LSTM  networks 
applied to Sentiment Analysis.


Adaptation, corrections, and modifications by Mauro Benetti 03-2021

This article was based on:

* https://towardsdatascience.com/nlp-sentiment-analysis-for-beginners-e7897f976897
* https://www.mdeditor.tw/pl/pISR/zh-hk
* https://stackabuse.com/removing-stop-words-from-strings-in-python/

For additional information you can visit:
* https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
* http://www.nltk.org/data.html



# 
# Introduction

**Natural Language Processing** (NLP) is the area of machine learning that focuses on the generation and understanding of language. Its main objective is to enable machines to understand, communicate and interact with humans in a natural way.

NLP has many tasks such as **Text Generation**, **Text Classification**,**Machine Translation**,  **Speech Recognition**, **Sentiment Analysis**, etc. For a beginner to NLP, looking at these tasks and all the techniques involved in handling such tasks can be quite daunting. 
And in fact, it is very difficult for a newbie to know exactly where and how to start.

Out of all the NLP tasks, Sentiment Analysis (SA) is probably one of the easiest, which makes it the most suitable starting point for anyone who wants to start going into NLP.

In this article, various techniques were compiled to perform SA, ranging from simple ones like **TextBlob** and **NLTK** to more advanced ones like **Sklearn** and **Long Short Term Memory (LSTM)** networks.

After reading this, you can expect to understand the followings:

*   Toolkits used in SA: TextBlob and NLTK
*   Algorithms used in SA: Naive Bayes, SVM, Logistic Regression and LSTM
*   Jargons like stop-word removal, stemming, bag of words, corpus, tokenization etc.
*   Create a word cloud.

The flow of this article:

*   Data cleaning and pre-processing
*   TextBlob
*   Algorithms: Logistic Regression, Naive Bayes, SVM and LSTM
    
## Problem 

In this article, we will work with a data set that consists of 3000 sentences coming from reviews on imdb.com, amazon.com, and yelp.com. Each sentence is labelled according to whether it comes from a positive review (labelled as 1) or a negative review (labelled as 0). The folder sentiment_labelled_sentences(containing the data file full_set.txt) should be in the same directory as your notebook/script.

## Content of this repository

Included in this repository, the file '01_Preparation.py' is written to be executed in 
VS Code with an interactive python session. A jupyter notebook and a markdown version of 
the notebook is included due to the convenience of reading like a book. 

Datasets in this repo: 
This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al, KDD 2015. Please cite the paper if you want to use it.

It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

Format:     "sentence" \t score \n
Details:    Score, either 1 (for positive) or 0 (for negative)	

The sentences come from three different websites and fields:

* imdb.com
* amazon.com
* yelp.com

For each website, there are 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. The goal is to select sentences that have a positive or negative connotation, with almost no neutral sentences to be selected.

**For the full datasets please visit:**

* IMDB: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
* amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: understanding rating dimensions with review text'
* yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge

## Advantages of running on VS Code instead of a Jupyter Notebook

The following tutorial explains the VS Code Interactive Python feature:
https://code.visualstudio.com/docs/python/jupyter-support-py#_python-interactive-window

This is a game-changer for data analysis because you no longer need to code in a Jupyter Notebook to execute your analysis. Simply write your code in a .py file and press Shift+ENTER to execute line-by-line in the Python Interactive Window. 

Repeat this process as you run code, explore, and build out your analysis. Note that you can also type Python directly into the Interactive Window just like you can type directly in R’s Console as well to execute code. Also, It is possible to see all the variables created and their type

**With this approach we avoid the limitations presented by Jupyter Notebooks when you want to do Version control (eg: Git).**
co
<p align="center">
  <img width="900" height="500" src="figures\Capture.PNG">
</p>


For more information visit (https://code.visualstudio.com/docs/python/jupyter-support-py)

<p align="center">
  <img width="460" height="300" src="https://code.visualstudio.com/assets/docs/python/jupyter/plot-viewer.gif">
</p>

