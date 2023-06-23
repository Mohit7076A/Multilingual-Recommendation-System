from main import final_recommender, remove_endline, string_to_list
import pandas as pd
import numpy as np
import pickle

sessions = pd.read_csv('Dataset MLRS/sessions_train.csv')
sessions = sessions.fillna("")

UK_sessions = sessions[sessions['locale'] == 'UK'].drop(['locale'], axis=1)
UK_sessions = UK_sessions.reset_index(drop=True)
UK_sessions['prev_items'] = UK_sessions['prev_items'].apply(string_to_list)
UK_sessions['prev_items'] = UK_sessions['prev_items'].apply(remove_endline)

DE_sessions = sessions[sessions['locale'] == 'DE'].drop(['locale'], axis=1)
DE_sessions = DE_sessions.reset_index(drop=True)
DE_sessions['prev_items'] = DE_sessions['prev_items'].apply(string_to_list)
DE_sessions['prev_items'] = DE_sessions['prev_items'].apply(remove_endline)

JP_sessions = sessions[sessions['locale'] == 'JP'].drop(['locale'], axis=1)
JP_sessions = JP_sessions.reset_index(drop=True)
JP_sessions['prev_items'] = JP_sessions['prev_items'].apply(string_to_list)
JP_sessions['prev_items'] = JP_sessions['prev_items'].apply(remove_endline)

FR_sessions = sessions[sessions['locale'] == 'FR'].drop(['locale'], axis=1)
FR_sessions = FR_sessions.reset_index(drop=True)
FR_sessions['prev_items'] = FR_sessions['prev_items'].apply(string_to_list)
FR_sessions['prev_items'] = FR_sessions['prev_items'].apply(remove_endline)

IT_sessions = sessions[sessions['locale'] == 'IT'].drop(['locale'], axis=1)
IT_sessions = IT_sessions.reset_index(drop=True)
IT_sessions['prev_items'] = IT_sessions['prev_items'].apply(string_to_list)
IT_sessions['prev_items'] = IT_sessions['prev_items'].apply(remove_endline)

ES_sessions = sessions[sessions['locale'] == 'ES'].drop(['locale'], axis=1)
ES_sessions = ES_sessions.reset_index(drop=True)
ES_sessions['prev_items'] = ES_sessions['prev_items'].apply(string_to_list)
ES_sessions['prev_items'] = ES_sessions['prev_items'].apply(remove_endline)

# print(final_recommender(UK_sessions['prev_items'][0],'UK'))
UK_predicted = UK_sessions['prev_items'].apply(lambda x: final_recommender(x, 'UK'))
pickle.dump(UK_predicted, open('UK_predicted.pkl', 'wb'))
del UK_predicted

DE_predicted = DE_sessions['prev_items'].apply(lambda x: final_recommender(x, 'DE'))
pickle.dump(DE_predicted, open('DE_predicted.pkl', 'wb'))
del DE_predicted

JP_predicted = JP_sessions['prev_items'].apply(lambda x: final_recommender(x, 'JP'))
pickle.dump(JP_predicted, open('JP_predicted.pkl', 'wb'))
del JP_predicted

FR_predicted = FR_sessions['prev_items'].apply(lambda x: final_recommender(x, 'FR'))
pickle.dump(FR_predicted, open('FR_predicted.pkl', 'wb'))
del FR_predicted

IT_predicted = IT_sessions['prev_items'].apply(lambda x: final_recommender(x, 'IT'))
pickle.dump(IT_predicted, open('IT_predicted.pkl', 'wb'))
del IT_predicted

ES_predicted = ES_sessions['prev_items'].apply(lambda x: final_recommender(x,'ES'))
pickle.dump(ES_predicted, open('ES_predicted.pkl', 'wb'))
del ES_predicted


# def accuracy(predicted, next):
#     sz = predicted.shape[0]
#     count = 0
#     for i in range(0, sz):
#         item = next[i]
#         for j in predicted[i]:
#             if j == item:
#                 count += 1
#                 break
#     return count/sz

# print(accuracy(UK_sessions['predicted_products'],UK_sessions['next_item']))

