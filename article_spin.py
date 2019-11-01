import nltk
import random
import numpy as np 
from bs4 import BeautifulSoup

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(),features="lxml")
positive_reviews = positive_reviews.findAll('review_text')
#print(positive_reviews[1])
tri={}

# Setup the dictionary
for review in positive_reviews:
    token= review.text.lower()
    token= nltk.tokenize.word_tokenize(token)
    for i in range(len(token)-2):
        ky=(token[i],token[i+2])
        if ky not in tri:
            tri[ky]=[]
        tri[ky].append(token[i+1])
print(tri)
