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

**Problem with this** : Dataset is very huge. Only UK alone has 5 lakh products, using countvectorizer with 10000 features mean to allocate upto 1TB of memory which is impossible for normal setups. Even if the vectorization is done using less features, the cosine matrix has to N x N where N is the rows and moreover all processes could take days to train.
To overcome these, I used HashingVectorizer for converting words into vectors, the idea behind  it is very memory efficient and stores only those indices which have values, as the resulting vector resembles sparse matrix.


# PROJECT  
I used PyCharm for this project. Also my colab and jupyter crashed many times while handling big data.

**1. Importing Products**

All locales products contained in a single file products_train.csv They are extracted using Pandas library. 
Thereafter I filtered out the product based on their locales. As most of the fields are empty , I filled up with empty string and reset all indices staring from zero. Dropping locale as it wont be needed and prices as it wont be helpful in my approach.![Screenshot (108)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/3ed1de93-aa40-4b6b-90d3-abad73e1eea8)

**2. Data Preprocessing**

Techniques I have used to preprocess data:

A. Removing spaces: for columns size, color, author and brand (to differetiate between common name/sizes like '5' cm seems similar to '5' km for machine but '5cm' and '5km' isnt.

B. Remove punctation marks

C. Removing Stopwords

![Screenshot (112)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/d9a6a3c7-9b0d-4ad0-be82-44ef0cb55342)

D. Using Stemming of words : For non english words, I used SnowBall Stemmer library

In sesssions files, the products are in string, to convert them into list:

E. String to list

![Screenshot (109)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/b17b06a5-ae91-44e3-b46a-40a399449bc7)

F. Combining all the columns to one single column and drop other ones:

![Screenshot (110)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/9afe3bb8-3c55-44b3-a5b1-e71564157f0c)
G. Lowercasing all the letters in that resultant string (exeception - for japanese and german language)

After all those operations the dataset looks like this for UK products:
![Screenshot (111)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/b0b6ff07-ef46-42e2-8682-bbde655a18b4)

Above are applied to other locales too.


**3. Vectorization and Model**

We got string of words corresponding to each product. Now my approach was to create vectors for each each string word and count them which is automatically done thorugh Vectorizers functions in sklearn.feature_extraction. First I used countVectorizer which was a bad decision as it took up all the memory and my system crashed. Then after further searching I got to know about Hashing and TF-IDF vectorizer which are better than it.

I used HashingVectorizer as it is super memory efficient for large datasets and store the resulting vector because I dont want to run again for 2-3 hours
and imported it using pickle library.
![Screenshot (113)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/3a183c68-dd15-4533-b78c-6254493749b4)

Similarly for other locales, all vectors are stored and imported for further modelling.

I wanted to knothe similar vectors for a given vector, it could be found using euclidean distance, manahattan distane or cosine distance. As first two are not recommeded for large datasets, I decided to use cosine distance but again the problem of memory comes and I didnt fing any optimized way to optimize this. To tackle this issue I make model that would take up the vector and iterate through all vectors to find similar vectors that would take O(N) space instead of storing all the similarity scores of each vector in O(NxN) space. The problem with this is that every time I run this model, it will take much time even for predicting for one list of item however I didnt find any solution to it.

![Screenshot (114)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/53729a27-dd09-466a-995e-e0c2169291da)

This function takes one productId and its local; and returns 5 (I printed first 5 at that time instead of 10)  similar productIDs.
![Screenshot (101)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/e8298ae9-7228-46a1-ba7f-837205b65ef1)

After preprocessing the sessions data, 'prev_items' column has list of productIDs.

My next approach was to select the similar products from last 5 items purchased by user as what I have experienced while surfing E-Commerce websites is the recommendations related to last 1-3 items and rarely saw product related to that I bought before 50 items. And assuming user will not buy the same product again exists in the prev_item list.

Here are the result for one list from sessions:


![Screenshot (102)](https://github.com/Mohit7076A/Multilingual-Recommendation-System/assets/98163995/ec4b6924-75cf-406a-8f60-eb3fcd931977)


# Scope
Though I made a model which take some values of products and return similar products, but there is the need to make model for optimized like the similarity matrix which is taking enormous time to evaluate.

Also due to this, I wasnt able to calculate the accuracy of the sessions trained.
So that could also be optimized in future.
After knowing this, some other techniques/alogrithms can be used to improve the accuracy further.
Concluding this, since my deadline of this project is just around the corner and I could not make sufficient changes to the model , I finished up working on this project for now.
Through this project, I learnt many new things like about recommender systems, how these works, where are they used, I learnt new libraries to use and in future to I wish to learn more
Hope you like it!😊✨










