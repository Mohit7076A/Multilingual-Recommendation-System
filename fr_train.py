import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

#Stemming
from nltk.stem import SnowballStemmer
st = SnowballStemmer('french')
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
        if word not in set(stopwords.words("french")):
            y.append(word)

     return " ".join(y)


products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

FR_products = products[products['locale'] == 'FR'].drop(['locale', 'price'], axis=1)
print(FR_products.shape)
FR_products=FR_products.reset_index(drop=True)
print(FR_products.shape)
FR_products['size'] = FR_products['size'].apply(lambda x: x.replace(' ', ''))
FR_products['author'] = FR_products['author'].apply(lambda x: x.replace(' ', ''))
FR_products['brand'] = FR_products['brand'].apply(lambda x: x.replace(' ', ''))
FR_products['color'] = FR_products['color'].apply(lambda x: x.replace(' ', ''))

FR_products['tags'] = FR_products['title']+' ' + FR_products['brand']+' ' + FR_products['color']+' ' + FR_products['size'] + ' ' + FR_products['model']+' ' + FR_products['material']+' ' + FR_products['author']+' ' + FR_products['desc']

FR_products['tags'] = FR_products['tags'].apply(stem)
FR_products['tags'] = FR_products['tags'].apply(remove_punc)
FR_products['tags'] = FR_products['tags'].apply(remove_stop_words)
print(FR_products.iloc[0].id)

vector = HashingVectorizer(n_features=20000, norm=None, alternate_sign=False).fit_transform(FR_products['tags'])
print(vector)

import pickle
pickle.dump(vector,open('Vectors/FR_vector.pkl','wb'))

print(FR_products.head(10))