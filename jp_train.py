import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

import string

#Removing punctuation marks
def remove_punc(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


import advertools as adv
#Removing stop words
def remove_stop_words(text):
     y = []
     for word in text.split():
        if word not in set(adv.stopwords['japanese']):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

JP_products = products[products['locale'] == 'JP'].drop(['locale', 'price'], axis=1)
print(JP_products.shape)
JP_products=JP_products.reset_index(drop=True)
print(JP_products.shape)
JP_products['size'] = JP_products['size'].apply(lambda x: x.replace(' ', ''))
JP_products['author'] = JP_products['author'].apply(lambda x: x.replace(' ', ''))
JP_products['brand'] = JP_products['brand'].apply(lambda x: x.replace(' ', ''))
JP_products['color'] = JP_products['color'].apply(lambda x: x.replace(' ', ''))

JP_products['tags'] = JP_products['title']+' ' + JP_products['brand']+' ' + JP_products['color']+' ' + JP_products['size'] + ' ' + JP_products['model']+' ' + JP_products['material']+' ' + JP_products['author']+' ' + JP_products['desc']

JP_products['tags'] = JP_products['tags'].apply(remove_punc)
JP_products['tags'] = JP_products['tags'].apply(remove_stop_words)
print(JP_products.iloc[0].id)

vector = HashingVectorizer(n_features=100000, norm=None, alternate_sign=False).fit_transform(JP_products['tags'])
print(vector)


import pickle
pickle.dump(vector,open('Vectors/JP_vector.pkl','wb'))

print(JP_products.head(10))