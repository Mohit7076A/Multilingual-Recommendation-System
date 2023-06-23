import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

#Stemming
from nltk.stem import SnowballStemmer
st = SnowballStemmer('italian')
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
        if word not in set(stopwords.words("italian")):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

IT_products = products[products['locale'] == 'IT'].drop(['locale', 'price'], axis=1)
print(IT_products.shape)
IT_products=IT_products.reset_index(drop=True)
print(IT_products.shape)
IT_products['size'] = IT_products['size'].apply(lambda x: x.replace(' ', ''))
IT_products['author'] = IT_products['author'].apply(lambda x: x.replace(' ', ''))
IT_products['brand'] = IT_products['brand'].apply(lambda x: x.replace(' ', ''))
IT_products['color'] = IT_products['color'].apply(lambda x: x.replace(' ', ''))

IT_products['tags'] = IT_products['title']+' ' + IT_products['brand']+' ' + IT_products['color']+' ' + IT_products['size'] + ' ' + IT_products['model']+' ' + IT_products['material']+' ' + IT_products['author']+' ' + IT_products['desc']

IT_products['tags'] = IT_products['tags'].apply(stem)
IT_products['tags'] = IT_products['tags'].apply(remove_punc)
IT_products['tags'] = IT_products['tags'].apply(remove_stop_words)
print(IT_products.iloc[0].id)

vector = HashingVectorizer(n_features=20000, norm=None, alternate_sign=False).fit_transform(IT_products['tags'])
print(vector)


import pickle
pickle.dump(vector,open('Vectors/IT_vector.pkl','wb'))

print(IT_products.head(10))