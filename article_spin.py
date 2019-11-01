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
#print(tri)
# Set Probabilities 
tri_pro={}
for k,words in tri.items():
    
    if len(set(words)) > 1:   #Consider Cases when multiple occurences of each pair 
        #print(words,"**")
        d={}    # make new dictionary to hold words
        n=0
        for w in words:
            if w not in d:
                d[w]= 0
            d[w]+=1
            n+=1
        #print(d)
        for w,c in d.items():
            d[w]= float(c)/n
        tri_pro[k]=d
#print(tri_pro)

# define  a random sample of the words
def random_sample(d):
    r=random.random()
    cumulative=0
    for w,p in d.items():
        cumulative = p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(positive_reviews)
    token= review.text.lower()
    print("Original :",token)
    token= nltk.tokenize.word_tokenize(token)
    for i in range(len(token)-2):
        ky=(token[i],token[i+2])
        if ky in tri_pro:
            w=random_sample(tri_pro[ky]) # select random word from bunch
            token[i+1]=w
            #print(token)
    print(token)
    token = [k for k in token if k!= None]
    print("Spun:")
    print(" ".join(token).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))
        
test_spinner()
