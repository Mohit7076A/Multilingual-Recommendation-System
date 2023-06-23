import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

#Stemming
from nltk.stem import SnowballStemmer
st = SnowballStemmer('spanish')
def stem(text):
  y=[]
  for i in text.split():
    y.append(st.stem(i))
  return " ".join(y)

import string

#Removing punctuation marks
def remove_punc(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


from nltk.corpus import stopwords
nltk.download('stopwords')

#Removing stop words
def remove_stop_words(text):
     y = []
     for word in text.split():
        if word not in set(stopwords.words("spanish")):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

ES_products = products[products['locale'] == 'ES'].drop(['locale', 'price'], axis=1)
print(ES_products.shape)
ES_products=ES_products.reset_index(drop=True)
print(ES_products.shape)
ES_products['size'] = ES_products['size'].apply(lambda x: x.replace(' ', ''))
ES_products['author'] = ES_products['author'].apply(lambda x: x.replace(' ', ''))
ES_products['brand'] = ES_products['brand'].apply(lambda x: x.replace(' ', ''))
ES_products['color'] = ES_products['color'].apply(lambda x: x.replace(' ', ''))

ES_products['tags'] = ES_products['title']+' ' + ES_products['brand']+' ' + ES_products['color']+' ' + ES_products['size'] + ' ' + ES_products['model']+' ' + ES_products['material']+' ' + ES_products['author']+' ' + ES_products['desc']

ES_products['tags'] = ES_products['tags'].apply(stem)
ES_products['tags'] = ES_products['tags'].apply(remove_punc)
ES_products['tags'] = ES_products['tags'].apply(remove_stop_words)
print(ES_products.iloc[0].id)

vector = HashingVectorizer(n_features=20000, norm=None, alternate_sign=False).fit_transform(ES_products['tags'])
print(vector)

import pickle
pickle.dump(vector, open('Vectors/ES_vector.pkl', 'wb'))

print(ES_products.head(10))