import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

#Stemming
ps = PorterStemmer()
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
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
        if word not in set(stopwords.words("english")):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

UK_products = products[products['locale'] == 'UK'].drop(['locale', 'price'], axis=1)
print(UK_products.shape)
UK_products=UK_products.reset_index(drop=True)
print(UK_products.shape)
UK_products['size'] = UK_products['size'].apply(lambda x: x.replace(' ', ''))
UK_products['author'] = UK_products['author'].apply(lambda x: x.replace(' ', ''))
UK_products['brand'] = UK_products['brand'].apply(lambda x: x.replace(' ', ''))
UK_products['color'] = UK_products['color'].apply(lambda x: x.replace(' ', ''))

UK_products['tags'] = UK_products['title']+' ' + UK_products['brand']+' ' + UK_products['color']+' ' + UK_products['size'] + ' ' + UK_products['model']+' ' + UK_products['material']+' ' + UK_products['author']+' ' + UK_products['desc']

UK_products['tags'] = UK_products['tags'].apply(lambda x:x.lower())
UK_products['tags'] = UK_products['tags'].apply(stem)
UK_products['tags'] = UK_products['tags'].apply(remove_punc)
UK_products['tags'] = UK_products['tags'].apply(remove_stop_words)
print(UK_products.iloc[0].id)

vector = HashingVectorizer(n_features=100000, norm=None, alternate_sign=False).fit_transform(UK_products['tags'])
print(vector)

import pickle
pickle.dump(vector, open('Vectors/UK_vector.pkl', 'wb'))

print(UK_products.head(10))



