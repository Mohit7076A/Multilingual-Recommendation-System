import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from model import recommender
import string


# Prev_items are given as strings so the function to convert them into list
def string_to_list(text):
  text = text.translate(str.maketrans('', '', string.punctuation))
  return list(text.split(' '))


# Prev_items have endline character after splitting , function to remove this
def remove_endline(products):
  l = []
  for i in products:
    if (i[len(i)-1]) == '\n':
      i = i[:-1]
    l.append(i)
  return l


# Importing sessions and products
products = pd.read_csv('Dataset MLRS/products_train.csv')
sessions = pd.read_csv('Dataset MLRS/sessions_train.csv')

products = products.fillna("")
sessions = sessions.fillna("")

UK_products = products[products['locale'] == 'UK']['id']
UK_products = UK_products.reset_index(drop=True)
UK_sessions = sessions[sessions['locale'] == 'UK'].drop(['locale'], axis=1)
UK_sessions = UK_sessions.reset_index(drop=True)
UK_sessions['prev_items'] = UK_sessions['prev_items'].apply(string_to_list)
UK_sessions['prev_items'] = UK_sessions['prev_items'].apply(remove_endline)

DE_products = products[products['locale'] == 'DE']['id']
DE_products = DE_products.reset_index(drop=True)
DE_sessions = sessions[sessions['locale'] == 'DE'].drop(['locale'], axis=1)
DE_sessions = DE_sessions.reset_index(drop=True)
DE_sessions['prev_items'] = DE_sessions['prev_items'].apply(string_to_list)
DE_sessions['prev_items'] = DE_sessions['prev_items'].apply(remove_endline)

JP_products = products[products['locale'] == 'JP']['id']
JP_products = JP_products.reset_index(drop=True)
JP_sessions = sessions[sessions['locale'] == 'JP'].drop(['locale'], axis=1)
JP_sessions = JP_sessions.reset_index(drop=True)
JP_sessions['prev_items'] = JP_sessions['prev_items'].apply(string_to_list)
JP_sessions['prev_items'] = JP_sessions['prev_items'].apply(remove_endline)

FR_products = products[products['locale'] == 'FR']['id']
FR_products = FR_products.reset_index(drop=True)
FR_sessions = sessions[sessions['locale'] == 'FR'].drop(['locale'], axis=1)
FR_sessions = FR_sessions.reset_index(drop=True)
FR_sessions['prev_items'] = FR_sessions['prev_items'].apply(string_to_list)
FR_sessions['prev_items'] = FR_sessions['prev_items'].apply(remove_endline)

IT_products = products[products['locale'] == 'IT']['id']
IT_products = IT_products.reset_index(drop=True)
IT_sessions = sessions[sessions['locale'] == 'IT'].drop(['locale'], axis=1)
IT_sessions = IT_sessions.reset_index(drop=True)
IT_sessions['prev_items'] = IT_sessions['prev_items'].apply(string_to_list)
IT_sessions['prev_items'] = IT_sessions['prev_items'].apply(remove_endline)

ES_products = products[products['locale'] == 'ES']['id']
ES_products = ES_products.reset_index(drop=True)
ES_sessions = sessions[sessions['locale'] == 'ES'].drop(['locale'], axis=1)
ES_sessions = ES_sessions.reset_index(drop=True)
ES_sessions['prev_items'] = ES_sessions['prev_items'].apply(string_to_list)
ES_sessions['prev_items'] = ES_sessions['prev_items'].apply(remove_endline)

# Importing corresponding vectors
UK_vector = pickle.load(open('Vectors/UK_vector.pkl', 'rb'))
DE_vector = pickle.load(open('Vectors/DE_vector.pkl', 'rb'))
JP_vector = pickle.load(open('Vectors/JP_vector.pkl', 'rb'))
FR_vector = pickle.load(open('Vectors/FR_vector.pkl', 'rb'))
IT_vector = pickle.load(open('Vectors/IT_vector.pkl', 'rb'))
ES_vector = pickle.load(open('Vectors/ES_vector.pkl', 'rb'))

# print(UK_sessions.shape)
# print(DE_sessions.shape)
# print(JP_sessions.shape)
# print(ES_sessions.shape)
# print(FR_sessions.shape)
# print(IT_sessions.shape)



# As the previous products list size varies from 1 to hundreds, I take only last five times
def final_recommender(list_of_products,locale):
  list_of_products.reverse()
  similar_products=[]
  if len(list_of_products)>5:
    new_list = list_of_products[:5]
  else:
    new_list=list_of_products

  for i in new_list:
    temp_prod=recommender(i,locale)
    for j in temp_prod:
      if j not in list_of_products:
        similar_products.append(j)

  return similar_products

