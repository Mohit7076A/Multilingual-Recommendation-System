import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

#Stemming
from nltk.stem.snowball import GermanStemmer
st = GermanStemmer()
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
        if word not in set(stopwords.words("german")):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

DE_products = products[products['locale'] == 'DE'].drop(['locale', 'price'], axis=1)
print(DE_products.shape)
DE_products=DE_products.reset_index(drop=True)
print(DE_products.shape)
DE_products['size'] = DE_products['size'].apply(lambda x: x.replace(' ', ''))
DE_products['author'] = DE_products['author'].apply(lambda x: x.replace(' ', ''))
DE_products['brand'] = DE_products['brand'].apply(lambda x: x.replace(' ', ''))
DE_products['color'] = DE_products['color'].apply(lambda x: x.replace(' ', ''))

DE_products['tags'] = DE_products['title']+' ' + DE_products['brand']+' ' + DE_products['color']+' ' + DE_products['size'] + ' ' + DE_products['model']+' ' + DE_products['material']+' ' + DE_products['author']+' ' + DE_products['desc']

DE_products['tags'] = DE_products['tags'].apply(stem)
DE_products['tags'] = DE_products['tags'].apply(remove_punc)
DE_products['tags'] = DE_products['tags'].apply(remove_stop_words)
print(DE_products.iloc[0].id)

vector = HashingVectorizer(n_features=100000, norm=None, alternate_sign=False).fit_transform(DE_products['tags'])
print(vector)

import pickle
pickle.dump(vector,open('Vectors/DE_vector.pkl','wb'))

print(DE_products.head(10))