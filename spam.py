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
#print(corpus)

# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
X=cv.fit_transform(corpus).toarray()

#Dummy variables for spam and ham
y=pd.get_dummies(msg['label'])
y=y.iloc[:,1].values
#print(len(X),len(y))
#Create test and train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Training Model for classification

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
spam_detect= MultinomialNB().fit(X_train,y_train)

pred=spam_detect.predict(X_test)
print(confusion_matrix(pred,y_test))
print(accuracy_score(pred,y_test))

