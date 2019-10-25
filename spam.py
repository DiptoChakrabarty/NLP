import pandas as pd
msg = pd.read_csv("smsspamcollection/SMSSpamCollection",
                    sep='\t',names=['label','text'])

# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]
for i in range(len(msg)):
    