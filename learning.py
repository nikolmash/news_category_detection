#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from preprocessing import TextPreprocessing #это из файла preprocessing.py


data = pd.read_csv('lenta_ru_news.csv')


class ModelLearning:
    def __init__(self, is_stopwords, is_lemmatization):
        self.is_stopwords = is_stopwords
        self.is_lemmatization = is_lemmatization
    
    def preprocess(self, data, tfidf):
      prep = TextPreprocessing(self.is_stopwords, self.is_lemmatization)
      data['text'] = data['text'].apply(prep.preprocess)
      topic_to_id = {topic: id for id, topic in enumerate(data['topic'].unique())}
      data['topic'] = data['topic'].map(topic_to_id)
      texts = data['text']
      tfidf.fit(texts)
      return data, tfidf

    def learn(self, model, data, tfidf):
      target = data['topic']
      texts = data['text']
      train = tfidf_vectorizer.transform(texts)
      model.fit(train, target)
      return model

ml = ModelLearning(0,0) # 0 0 - параметры предобработки
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3)) # создаем тфидф
data, tfidf_vectorizer = ml.preprocess(data, tfidf_vectorizer) #предобработали данные и обучили тфидф
lr = LogisticRegression(solver='liblinear') #создали логрег
lr = ml.learn(lr, data, tfidf_vectorizer) # обучили лог рег
dump(lr, 'lr_0_0.joblib') # сохранили (0 0 - параметры предобработки)
dump(tfidf_vectorizer, 'tfidf_0_0.pkl') # сохранили тфидф (0 0 - параметры предобработки)

