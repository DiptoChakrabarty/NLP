# Latent Semantic Analysis
import nltk
import numpy as np 
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from  sklearn.decomposition import TruncatedSVD

lemm= WordNetLemmatizer()

titles = [line.rstrip() for line in open('book_title.txt')]

stopwords = set(w.rstrip() for w in stopwords.words('english'))

stopwords = stopwords.union({
    'introduction','edition','series','approach','card',
    'access','application','package','brief','vol','fundamental',
    'second','third','fourth','first','guide','essential','print'
})

def tokenize_words(s):
    s =s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>3] # remove short words
    tokens = [lemm.lemmatize(t) for t in tokens]
    tokens = [ t if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


