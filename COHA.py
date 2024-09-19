#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:40:12 2024

@author: tuituiwang
"""

import os
from SequentialEmbeddings import SequentialEmbedding
# dir(SequentialEmbedding)
from collections import defaultdict
import csv
import pandas as pd
# import importlib
# importlib.reload(Embeddings)

#load the word lists and store them as dictionary
def load_attrs(path):
    text = open(path, "r")
    text_list = list(set(text.read().lower().splitlines()))
    return text_list

def build_dict(w1, w2):
    category_dict = defaultdict(list)
    category_dict[w1] = load_attrs(os.path.join("./WordList/", w1 + ".txt"))
    category_dict[w2] = load_attrs(os.path.join("./WordList/", w2 + ".txt"))
    return category_dict

def get_bias_score(a_dict):
    bias_score_dict = defaultdict(float)
    for key, value in a_dict.items():
        bias_score_dict = embeddings_11.get_mean_diff_dict(a_dict[key], male_words, female_words)
    return bias_score_dict

# def save_to_csv_pvalues(csv_filename="p_values_changes.csv", num_executions=5):
#     # Open the CSV file in write mode initially to add headers
#     with open(csv_filename, mode='w', newline='') as csv_file:
#         csv_writer = None
        
#         for i in range(1, num_executions + 1):
#             # Execute the function and get the dictionary
            
            
#             # If it's the first run, write the header
#             if i == 1:
#                 result_dict_1 = embeddings_11.weat_rand_test_dict(B1_dict['career'], B1_dict['family'], male_words, female_words, 100000)
#                 csv_writer = csv.DictWriter(csv_file, fieldnames=[''] + list(result_dict_1.keys()))
#                 csv_writer.writeheader()
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_1)
#             elif i == 2:
#                 result_dict_2 = embeddings_11.weat_rand_test_dict(B2_dict['math'], B2_dict['arts'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_2)
#             elif i == 3:
#                 result_dict_3 = embeddings_11.weat_rand_test_dict(B3_dict['science'], B3_dict['arts'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_3)
#             elif i == 4:
#                 result_dict_4 = embeddings_11.weat_rand_test_dict(B4_dict['intelligence'], B4_dict['appearance'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_4)
#             elif i == 5:
#                 result_dict_5 = embeddings_11.weat_rand_test_dict(B5_dict['strong'], B5_dict['weak'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_5)
#             # Append the row to the CSV file
#             csv_writer.writerow(row)
#             print("p values calc, done!", i)
        
#         return result_dict_1, result_dict_2, result_dict_3, result_dict_4, result_dict_5
        

# def save_to_csv_d(csv_filename="d_changes.csv", num_executions=5):
#     # Open the CSV file in write mode initially to add headers
#     with open(csv_filename, mode='w', newline='') as csv_file:
#         csv_writer = None
        
#         for i in range(1, num_executions + 1):
#             # Execute the function and get the dictionary
            
            
#             # If it's the first run, write the header
#             if i == 1:
#                 result_dict_1 = embeddings_11.cohen_d_dict(B1_dict['career'], B1_dict['family'], male_words, female_words, 100000)
#                 csv_writer = csv.DictWriter(csv_file, fieldnames=[''] + list(result_dict_1.keys()))
#                 csv_writer.writeheader()
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_1)
#             elif i == 2:
#                 result_dict_2 = embeddings_11.cohen_d_dict(B2_dict['math'], B2_dict['arts'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_2)
#             elif i == 3:
#                 result_dict_3 = embeddings_11.cohen_d_dict(B3_dict['science'], B3_dict['arts'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_3)
#             elif i == 4:
#                 result_dict_4 = embeddings_11.cohen_d_dict(B4_dict['intelligence'], B4_dict['appearance'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_4)
#             elif i == 5:
#                 result_dict_5 = embeddings_11.cohen_d_dict(B5_dict['strong'], B5_dict['weak'], male_words, female_words, 100000)
#                 row = {'': f'b{i}'}
#                 row.update(result_dict_5)
#             # Append the row to the CSV file
#             csv_writer.writerow(row)
#             print("d calc, done!", i)
        
#         return result_dict_1, result_dict_2, result_dict_3, result_dict_4, result_dict_5
    


def write_csv(filename, a_dict):
    for field, a_list in a_dict.items():
        file_path = os.path.join(filename + field +'.csv')
        
        # Check if the file already exists
        file_exists = os.path.exists(file_path)
        
        with open(file_path, mode='a', newline='') as csv_file:
            for w in a_list:
                mean_diff_dict = embeddings_11.get_mean_diff_dict(w, male_words, female_words)
                
                # If csv_writer is None or if it's the first write, write the header
                csv_writer = csv.DictWriter(csv_file, fieldnames=[''] + list(mean_diff_dict.keys()))
                
                if not file_exists:
                    csv_writer.writeheader()
                    file_exists = True  # Ensure the header is only written once
                
                row = {'': w}
                row.update(mean_diff_dict)
                csv_writer.writerow(row)
                
def write_SC(dicts):
    os.makedirs('./SC_data/')
    filepath_ts = './SC_data/SC_ts.csv'
    filepath_d = './SC_data/SC_d.csv'
    filepath_p = './SC_data/SC_p.csv'
    
    field_ts = []
    field_d = []
    field_p = []
    
               
    def write_SC_ts(filepath_ts, a_dict):
        file_exists = os.path.exists(filepath_ts)
        results_dict_ts = {}
        with open(filepath_ts, mode = 'a', newline = '') as csv_file:
            for field, a_list in a_dict.items():
                if field not in field_ts:
                    stats = embeddings_11.SC_WEAT_ts_dict(a_list, male_words, female_words)
                    results_dict_ts[field] = stats
                    field_ts.append(field)
                    
                    csv_writer = csv.DictWriter(csv_file, fieldnames=['']+list(stats.keys()))
                    print(field + " calc done ts!")
                    
                    if not file_exists:
                        csv_writer.writeheader()
                        file_exists = True
                        
                    row = {'': field}
                    row.update(stats)
                    csv_writer.writerow(row)
             
        return results_dict_ts
    
    def write_SC_d(filepath_d, a_dict):
        file_exists = os.path.exists(filepath_d)
        results_dict_d = {}
        with open(filepath_d, mode = 'a', newline = '') as csv_file:
            for field, a_list in a_dict.items():
                if field not in field_d:
                    field_d.append(field)
                    stats = embeddings_11.SC_WEAT_d_dict(a_list, male_words, female_words)
                    results_dict_d[field] = stats
                    
                    csv_writer = csv.DictWriter(csv_file, fieldnames=['']+list(stats.keys()))
                    print(field + " calc done d!")
                    
                    if not file_exists:
                        csv_writer.writeheader()
                        file_exists = True
                        
                    row = {'': field}
                    row.update(stats)
                    csv_writer.writerow(row)
             
        return results_dict_d
    
    def write_SC_p(filepath_p, a_dict):
        file_exists = os.path.exists(filepath_p)
        results_dict_p = {}
        with open(filepath_p, mode = 'a', newline = '') as csv_file:
            for field, a_list in a_dict.items():
                if field not in field_p:
                    field_p.append(field)
                    stats = embeddings_11.SC_WEAT_p_dict(a_list, male_words, female_words, 100000)
                    results_dict_p[field] = stats
                    
                    csv_writer = csv.DictWriter(csv_file, fieldnames=['']+list(stats.keys()))
                    print(field + " calc done p!")
                    
                    if not file_exists:
                        csv_writer.writeheader()
                        file_exists = True
                        
                    row = {'': field}
                    row.update(stats)
                    csv_writer.writerow(row)
             
        return results_dict_p



    #process_ts
    for d in dicts:
        write_SC_ts(filepath_ts, d)
        write_SC_d(filepath_d, d)
        write_SC_p(filepath_p, d)
        
    
def write_clusters(a_list):
    df = pd.DataFrame()
    for w in a_list:
        cluster_dict = embeddings_11.get_closet(w)
        row_df = pd.DataFrame([cluster_dict], index = [w])
        df = pd.concat([df, row_df])
        
            
    return df
            
            
        
'''
Testing section
'''

# test_lists = ["poetry" , "art", 
# "shakespeare",
# "dance",
# "literature",
# "novel",
# "symphony",
# "drama"]


# test_embeddings = SequentialEmbedding.load("/Users/tuituiwang/Documents/PG/diss/Methodology/Data/COHA_word_sgns", range(1850, 1870, 10))
# print(test_embeddings)

# print(list(range(1900, 2010, 10)))
# print(len(list(range(1900, 2010, 10))))

# #test: calculate the mean of similarity over time
# time_sims_mean_test = test_embeddings.get_time_sims_wordlists("women", test_lists)
# print(time_sims_mean_test)



'''
Complete data manipulation process
'''

#load the embeddings (normalized by default)
embeddings_11 = SequentialEmbedding.load("/Users/tuituiwang/Documents/PG/diss/Methodology/Data/COHA_word_sgns", range(1900, 2010, 10))

#B1: career vs family
B1_dict = build_dict('career', 'family')
# B2: maths vs arts
B2_dict = build_dict('math', 'arts')
# B3: science vs arts
B3_dict = build_dict('science', 'arts')
# B4: intelligence vs appearance
B4_dict = build_dict('intelligence', 'appearance')
# B5: strength vs weakness 
B5_dict = build_dict('strong', 'weak')

#load male and female word list
female_words = load_attrs("./WordList/female_terms.txt")
male_words = load_attrs("./WordList/male_terms.txt")

#B1:
# B1_p = embeddings_11.weat_rand_test_dict(B1_dict['career'], B1_dict['family'], male_words, female_words)
# B1_d = embeddings_11.cohen_d_dict(B1_dict['career'], B1_dict['family'], male_words, female_words)
# B1_bias_score = get_bias_score(B1_dict)

## WEAT:
# d_dict_1, d_dict_2, d_dict_3, d_dict_4, d_dict_5 = save_to_csv_d()
# p_value_dict_1, p_value_dict_2, p_value_dict_3, p_value_dict_4, p_value_dict_5 = save_to_csv_pvalues("p_values_changes.csv", 5)

# write_csv("B1", B1_dict)
# write_csv("B2", B2_dict)
# write_csv("B3", B3_dict)
# write_csv("B4", B4_dict)
# write_csv("B5", B5_dict)


#SC-WEAT
dicts = [B1_dict, B2_dict, B3_dict, B4_dict, B5_dict]
write_SC(dicts)

# male_closest = write_clusters(male_words)
# male_closest.to_excel('male_closest.xlsx')
# female_closest = write_clusters(female_words)
# female_closest.to_excel('female_closest.xlsx')


