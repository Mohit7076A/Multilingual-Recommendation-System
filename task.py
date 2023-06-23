import pandas as pd
import numpy as np
from main import final_recommender, remove_endline, string_to_list


# ************************* TASK 1 *********************************
s1 = pd.read_csv('Dataset MLRS/sessions_test_task1.csv')

UK_1 = s1[s1['locale'] == 'UK']
UK_1['prev_items'].apply(string_to_list)
UK_1['prev_items'].apply(remove_endline)


DE_1 = s1[s1['locale'] == 'DE']
DE_1['prev_items'].apply(string_to_list)
DE_1['prev_items'].apply(remove_endline)

JP_1 = s1[s1['locale'] == 'JP']
JP_1['prev_items'].apply(string_to_list)
JP_1['prev_items'].apply(remove_endline)

# Results
UK_predicted = final_recommender(UK_1['prev_items'], 'UK')
DE_predicted = final_recommender(DE_1['prev_items'], 'DE')
JP_predicted = final_recommender(JP_1['prev_items'], 'JP')

# ************************* TASK 2 ***********************************
s2 = pd.read_csv('Dataset MLRS/sessions_test_task2.csv')

ES_2 = s2[s2['locale'] == 'ES']
ES_2['prev_items'].apply(string_to_list)
ES_2['prev_items'].apply(remove_endline)


FR_2 = s2[s2['locale'] == 'FR']
FR_2['prev_items'].apply(string_to_list)
FR_2['prev_items'].apply(remove_endline)

IT_2 = s2[s2['locale'] == 'IT']
IT_2['prev_items'].apply(string_to_list)
IT_2['prev_items'].apply(remove_endline)

# Results
ES_predicted = final_recommender(ES_2['prev_items'], 'ES')
FR_predicted = final_recommender(FR_2['prev_items'], 'FR')
IT_predicted = final_recommender(IT_2['prev_items'], 'IT')
