# Latent Semantic Analysis
import nltk
import numpy as np 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
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
    tokens = [ t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

word_index_map ={}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

for title in titles:
    try:
        #title= title.encode('ascii','ignore').decode('utf-8')
        all_titles.append(title)
        tokens = tokenize_words(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token]=current_index
                current_index+=1
                index_word_map.append(token)
    except:
        pass

#print(word_index_map)



def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        #print(i)
        x[i]=1
   # print(x)
    
    return x

N=len(all_tokens)
D=len(word_index_map)
X=np.zeros((D,N))
i=0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i+=1
#print(X.shape)
#print(X)

svd = TruncatedSVD()
Z= svd.fit_transform(X)
plt.scatter(Z[:,0],Z[:,1])
for i in range(D):
    plt.annotate(s=index_word_map[i],xy=(Z[i,0],Z[i,1]))
plt.show()

