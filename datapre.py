import numpy as np
import pandas as pd

products = pd.read_csv('Dataset MLRS/products_train.csv')
print(products.shape)

products = products.fillna("")

UK_products = products[products['locale'] == 'UK'].drop(['locale', 'price'], axis=1)
DE_products = products[products['locale'] == 'DE'].drop(['locale', 'price'], axis=1)
JP_products = products[products['locale'] == 'JP'].drop(['locale', 'price'], axis=1)
FR_products = products[products['locale'] == 'FR'].drop(['locale', 'price'], axis=1)
IT_products = products[products['locale'] == 'IT'].drop(['locale', 'price'], axis=1)
ES_products = products[products['locale'] == 'ES'].drop(['locale', 'price'], axis=1)
print(UK_products.shape)
print(DE_products.shape)
print(JP_products.shape)
print(IT_products.shape)
print(FR_products.shape)
print(ES_products.shape)

