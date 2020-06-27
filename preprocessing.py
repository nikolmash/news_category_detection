#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk
import pymorphy2
from string import punctuation
from nltk.corpus import stopwords


class TextPreprocessing:
    def __init__(self, is_stopwords, 
                 is_lemmatization):
        self.is_stopwords = is_stopwords
        self.is_lemmatization = is_lemmatization
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = set(stopwords.words('russian'))
        self.punkt = punctuation + '«»—…“”*№–'
    
    def preprocess(self, text):
        clean_text = ' '.join([word.strip(self.punkt) for word in text.lower().split()])
        if self.is_stopwords == 1:
            clean_text = ' '.join([word for word in clean_text.split() if word not in self.stopwords])
        if self.is_lemmatization == 1:
            clean_text = ' '.join([self.morph.parse(word)[0].normal_form for word in clean_text.split()])
        return clean_text

