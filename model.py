import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Importing Products Indices
products = pd.read_csv('Dataset MLRS/products_train.csv')

products = products.fillna("")

UK_products = products[products['locale'] == 'UK']['id']
UK_products = UK_products.reset_index(drop=True)

DE_products = products[products['locale'] == 'DE']['id']
DE_products = DE_products.reset_index(drop=True)

JP_products = products[products['locale'] == 'JP']['id']
JP_products = JP_products.reset_index(drop=True)

FR_products = products[products['locale'] == 'FR']['id']
FR_products = FR_products.reset_index(drop=True)

IT_products = products[products['locale'] == 'IT']['id']
IT_products = IT_products.reset_index(drop=True)

ES_products = products[products['locale'] == 'ES']['id']
ES_products = ES_products.reset_index(drop=True)

# Importing vectors of different languages
UK_vector = pickle.load(open('Vectors/UK_vector.pkl', 'rb'))
DE_vector = pickle.load(open('Vectors/DE_vector.pkl', 'rb'))
JP_vector = pickle.load(open('Vectors/JP_vector.pkl', 'rb'))
FR_vector = pickle.load(open('Vectors/FR_vector.pkl', 'rb'))
IT_vector = pickle.load(open('Vectors/IT_vector.pkl', 'rb'))
ES_vector = pickle.load(open('Vectors/ES_vector.pkl', 'rb'))


# Getting index of the given product
def get_index(product_id, locale_products):
   size = locale_products.shape[0]
   for i in range(0, size):
    if locale_products[i] == product_id:
      return i


# Main recommender system


def recommender(product_id, locale):
    similar_items = []
    # Assigning vectors and products according to values given by user
    if locale == 'UK':
        vector = UK_vector
        locale_products = UK_products
    elif locale == 'DE':
        vector = DE_vector
        locale_products = DE_products
    elif locale == 'JP':
        vector = JP_vector
        locale_products = JP_products
    elif locale == 'FR':
        vector = FR_vector
        locale_products = FR_products
    elif locale == 'IT':
        vector = IT_vector
        locale_products = IT_products
    else:
        vector = ES_vector
        locale_products = UK_products

    index = get_index(product_id, locale_products)

    # Getting indices of top 5 similar products using cosine similarity
    similar_indices = np.flip(np.argsort(cosine_similarity(vector[index], vector)))

    for i in similar_indices[0][1:11]:
        similar_items.append(locale_products[i])

    return similar_items

# Model is ready for 5 recommendations

# productID = input('Enter the product ID')
# locale = input('Enter the locale/region')

# print(recommender(productID,locale))
