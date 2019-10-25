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
    review=re.sub('[^a-zA-Z]',' ',msg['text'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words("english"))]    
    review= " ".join(review)
    corpus.append(review)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
X=cv.fit_transform(corpus).toarray()