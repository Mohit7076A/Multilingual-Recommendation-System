# Multilingual-Recommendation-System
**Overview of the files** :
1. de_train : converting german products into vectors.
Similarly, for es_train, fr_train, it_train, jp_train, uk_train; all converting the respective locale products
2. model.py: the recommmender system for one productId
3. main.py: the recommender system for every list
4. Dataset link : https://drive.google.com/file/d/1utQ9lpcuWcRrLH5DZ3yEqDvhmhvRfvKa/view?usp=sharing
5. Vectors link (which are resulted by running code for hours) : https://drive.google.com/drive/folders/14vqdSbA7D2iydPVEPcY-8qHD5ku9kD2h?usp=drive_link
6. req.txt = python libraries/packages I used for the project


**Understanding the dataset:**

The dataset products_train.csv contains about 1.5 Million amazon products from 6 different locales - UK,DE and JP have more number of porducts (about 0.5 million) while FR,IT and ES have around 40-50K products.

Each product has 11 attritbutes or there are 11 columns in the dataset namely - id, locale, title, price, color, size, brand, model, material, author and description.![Screenshot (106)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/ed9a8821-2b3f-4e5f-8bcc-a5d1c76f80b1)
This project aims to recommend the next products first for the locale with more numbers of products - UK, DE, ES and then for IT, FR and ES given the list of previous products bought by the customer.

While the sessions_train.csv contains about 3.6 million rows with 3 columns - prev_items, next_item, locale![Screenshot (107)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/d8fd16ca-5061-4a44-9b32-810355fc1f61)


**Basic approach:**
At the very first, I thought of using NLP techniques to find the next product. However, I faced some problems regarding memory usage and time taken which I will discuss later. My basic approach to this was:- 
1. Import the datasets and filter accordingly to locales.
2. Using preprocessing data to the dataframes for natural language processing
3. Combining the result of all the columns (except the id) to one , lets say 'tags'
4. Vectorizing that one columns using simple class like CountVectorizer and find the similarity scores for all the items using cosine distances
5. Predicting the next items by comparing the cosine distances with prev items and recommend the closest ones.




