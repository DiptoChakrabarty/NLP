text="""All implied warranties, including without limitation, implied warranties of merchantability and fitness for a particular purpose, are limited to the duration of this limited warranty.
 In no event shall Samsung be liable for damages in excess of the purchase price of the product or for, without limitation, commercial loss of any sort; loss of use, time, data, reputation, 
 opportunity, goodwill, profits or savings; inconvenience; incidental, special, consequential or punitive damages; or damages arising from the use or inability to use the product. Some states 
 and jurisdictions do not allow limitations on how long an implied warranty lasts, or the disclaimer or limitation of incidental or consequential damages, so the above limitations and disclaimers may not apply to you.
Samsung makes no warranties or representations, express or implied, statutory or otherwise, as to the quality, capabilities, operations, performance or suitability of any third-party software or equipment used in 
conjunction with the product, or the ability to integrate any such software or equipment with the product, whether such third-party software or equipment is included with the product distributed by Samsung or otherwise.
 Responsibility for the quality, capabilities, operations, performance and suitability of any such third-party software or equipment rests solely with the user and the direct vendor, 
 owner or supplier of such third-party software or equipment."""
import nltk
# Cleaning the text 
import re
from nltk.corpus  import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
print("Packages imported")
ps=PorterStemmer()
lem=WordNetLemmatizer()
sentences=nltk.sent_tokenize(text)
#print(sentences)
corpus=[]
for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i]) #Removing all commas and exclamations etc
    #print(review)
    review=review.lower()
    review=review.split()
   #print(review)
    review=[lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    #print(review)
    review= ' '.join(review)
    corpus.append(review)

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv =TfidfVectorizer()
    X=cv.fit_transform(corpus).toarray()
    print(X)