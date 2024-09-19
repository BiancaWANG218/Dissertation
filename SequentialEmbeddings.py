#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:14:04 2024

@author: tuituiwang
"""

from Embeddings import Embedding
import collections
import numpy as np
import random

class SequentialEmbedding:
    def __init__(self, year_embeds, **kwargs):
        '''
        Purpose
        -------
        Initializes the SequentialEmbedding object with a dictionary of embeddings, 
        where the keys are years and the values are Embedding objects corresponding to those years.

        Parameters
        ----------
        year_embeds : OrderedDic
            where each key is a year and 
            each value is an Embedding object for that year.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        

        '''
        self.embeds = year_embeds
 
    @classmethod
    def load(cls, path, years, **kwargs):
        '''
        Purpose
        -------
        Creates an instance of SequentialEmbedding by loading embeddings 
        from files for a range of years.

        Parameters
        ----------
        path : 
            Base path where the embeddings are stored.
        years :
            A list of years for which embeddings need to be loaded.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        A SequentialEmbedding object initialized with embeddings loaded from files
        '''
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = Embedding.load(path + "/" + str(year), **kwargs)
        return SequentialEmbedding(embeds)

    def get_embed(self, year):
        '''
        Purpose
        -------
        The year for which to get the embedding

        Parameters
        ----------
        year

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return self.embeds[year]

    def get_subembeds(self, words, normalize=True):
        '''
        Creates sub-embeddings for a specific set of words from all embeddings in the sequence.

        Parameters
        ----------
        words : 
            A list of words for which sub-embeddings are to be created..
        normalize : optional
            Whether to normalize the sub-embeddings.. The default is True.

        Returns
        -------
        A new SequentialEmbedding object with sub-embeddings for the specified words.

        '''
        embeds = collections.OrderedDict()
        for year, embed in self.embeds.items():
            embeds[year] = embed.get_subembed(words, normalize=normalize)
        return SequentialEmbedding(embeds)

 
    def get_seq_neighbour_set(self, word, n=3):
        '''
        Retrieves a set of neighbors for a given word across all years, 
        combining results from each year’s embedding.

        Parameters
        ----------
        word :
            The word for which neighbors are found.
        n : optional
            Number of closest neighbors to consider per year. The default is 3.

        Returns
        -------
        neighbour_set :
            A set of unique neighbors across all years.

        '''
        neighbour_set = set([])
        for embed in self.embeds.itervalues():
            closest = embed.closest(word, n=n)
            for _, neighbour in closest:
                neighbour_set.add(neighbour)
        return neighbour_set

    def get_seq_closest(self, word, start_year, num_years=10, n=10):
        '''
        Finds the closest words to a given word over a sequence of years, 
        aggregating scores across years.

        Parameters
        ----------
        word: The word for which to find closest words.
        start_year: The starting year for the sequence.
        num_years: The number of years to consider.
        n: Number of closest words to return.


        Returns
        -------
        A list of the top n closest words across the specified years.

        '''
        closest = collections.defaultdict(float)
        for year in range(start_year, start_year + num_years, 10):
            embed = self.embeds[year]
            year_closest = embed.closest(word, n=n*10)
            for t in year_closest:
                closest[t[1]] += t[0]
        return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]
    
    def get_closet(self, word):
        closet_by_year = collections.OrderedDict()
        for year, embed in self.embeds.items():
            closet_by_year[year] =  embed.closest(word)
        
        return closet_by_year
    

    def get_word_subembeds(self, word, n=3, num_rand=None, word_list=None):
        '''
        Creates sub-embeddings for a specific set of words, 
        which can include both neighbors of a given word and additional random words if specified.

        Parameters
        ----------
        word: The base word to find neighbors.
        n: Number of closest neighbors to find.
        num_rand: Number of random words to include in the sub-embeddings.
        word_list: Predefined list of words for which sub-embeddings are created.

        Returns
        -------
        A SequentialEmbedding object with sub-embeddings for the specified words.

        '''
        if word_list == None:
            word_set = self.get_seq_neighbour_set(word, n=n)
            if num_rand != None:
                word_set = word_set.union(set(random.sample(self.embeds.values()[-1].iw, num_rand)))
            word_list = list(word_set)
        year_subembeds = collections.OrderedDict()
        for year,embed in self.embeds.items():
            year_subembeds[year] = embed.get_subembed(word_list)
        return SequentialEmbedding.from_ordered_dict(year_subembeds)
    
    def get_time_sims_singleword(self, word1, word2):
        '''
        Computes the similarity between two words for each year’s embedding.

        Parameters
        ----------
        word1, word2 : The words for which similarity is calculated.
        
        Returns
        -------
        time_sims : An OrderedDict with years as keys and similarity score as values.

        '''
        time_sims = collections.OrderedDict()
        for year, embed in self.embeds.items():
            time_sims[year] = embed.similarity(word1, word2)
        return time_sims
    
    def get_time_sims_wordlists(self, word, word_list):
        '''
        Computes the similarity between word and word_list

        Parameters
        ----------
        word : TYPE
            DESCRIPTION.
        word_list : List
            DESCRIPTION.

        Returns
        -------
        time_sims : An OrderedDict with years as keys and similarity scores (list) as values.

        '''
        # mean_time_sims = collections.OrderedDict()
        mean_time_sims = collections.OrderedDict()
        for year, embed in self.embeds.items():
            mean_time_sims[year] = np.mean([embed.similarity(word, w) for w in word_list])
        return mean_time_sims
    
    #WEAT in years
    
    def weat_rand_test_dict(self, target_list1, target_list2, attr_list1, attr_list2, iterations):
        p_values_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            p_values_dict[year] = embed.weat_rand_test(target_list1, target_list2, attr_list1, attr_list2, iterations)
        return p_values_dict
    
    def cohen_d_dict(self, target_list1, target_list2, attr_list1, attr_list2, iterations):
        cohen_d_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            cohen_d_dict[year] = embed.cohen_d(target_list1, target_list2, attr_list1, attr_list2, iterations)
        return cohen_d_dict
    
    def test_statistic_dict(self, target_list1, target_list2, attr_list1, attr_list2):
        test_stats_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            test_stats_dict[year] = embed.test_statistic(target_list1, target_list2, attr_list1, attr_list2)
        return test_stats_dict
    
    # def test_sum_dict(self, target_list1, target_list2, attr_list1, attr_list2):
    #     sum_1_dict = collections.OrderedDict() #key is year and sum is value
    #     sum_2_dict = collections.OrderedDict()
    #     for t in target_list1:
    #         diff_sims_t = self.get_mean_diff(t, attr_list1, attr_list2)
    #         for year, diff in diff_sims_t.items():
    #             sum_1_dict[year] += diff
        
    #     for t in target_list2:
    #         diff_sims_t = self.get_mean_diff(t, attr_list1, attr_list2)
    #         for year, diff in diff_sims_t.items():
    #             sum_2_dict[year] += diff
        
    #     return sum_1_dict, sum_2_dict
    
    def get_mean_diff_dict(self, w, list_1, list_2):
        #list_1 and list_2 are usually attr list
        mean_diff_years = collections.OrderedDict()
        for year, embed in self.embeds.items():
            mean_diff_years[year] = embed.get_mean_diff(w, list_1, list_2)
        return mean_diff_years
    
    #SC-WEAT
    def SC_WEAT_p_dict(self, target_list, attr_list1, attr_list2, iterations):
        SC_p_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            SC_p_dict[year] = embed.SC_WEAT_test(target_list, attr_list1, attr_list2, iterations)
        return SC_p_dict
    
    def SC_WEAT_d_dict(self, target_list, attr_list1, attr_list2):
        SC_d_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            SC_d_dict[year] = embed.SC_WEAT_d(target_list, attr_list1, attr_list2)
        return SC_d_dict
    
    def SC_WEAT_ts_dict(self, target_list, attr_list1, attr_list2):
        SC_ts_dict = collections.OrderedDict()
        for year, embed in self.embeds.items():
            SC_ts_dict[year] = embed.SC_WEAT_ts(target_list, attr_list1, attr_list2)
        return SC_ts_dict
        
    

# sum_1, sum_2 = self.test_sum_dict(target_list1, target_list2, attr_list1, attr_list2)
# years = set(sum_1.keys()).union(set(sum_2.kets()))
# test_stats_dict = {}
# for year in years:
#     value1 = sum_1.get(year, np.nan)
#     value2 = sum_2.get(year, np.nan)
#     if not np.isnan(value1) and not np.isnan(value2):
#         test_stats_dict[year] = value1 - value2
#     else:
#         test_stats_dict[year] = np.nan   

    

    
if __name__ == "__main__":
    def load_attrs(path):
        text = open(path, "r")
        text_list = list(set(text.read().lower().splitlines()))
        return text_list
    target_list_art = load_attrs("./WordList/arts.txt")
    target_list_math = load_attrs("./WordList/math.txt")
    female_list = load_attrs("./WordList/female_terms.txt")
    male_list = load_attrs("./WordList/male_terms.txt")
    
    test_seq_embeddings = SequentialEmbedding.load("./Data/COHA_word_sgns", range(1900, 1920, 10))
    # print("test_mean_diff", test_seq_embeddings.get_mean_diff_dict(target_list_art[0], male_list, female_list))
    # print("test_test_statistics_dict", test_seq_embeddings.test_statistic_dict(target_list_math, target_list_art, male_list, female_list))            
    # print("cohen d dict:", test_seq_embeddings.cohen_d_dict(target_list_math, target_list_art, male_list, female_list, 100000))
    # #print("p value dict:", test_seq_embeddings.weat_rand_test_dict(target_list_math, target_list_art, male_list, female_list, 100000))
    # print("SC_weat_ts_dict:", test_seq_embeddings.SC_WEAT_ts_dict(target_list_art, male_list, female_list))
    # print("SC_weat_d_dict:", test_seq_embeddings.SC_WEAT_d_dict(target_list_art, male_list, female_list))
    # print("SC_weat_p_dict:", test_seq_embeddings.SC_WEAT_p_dict(target_list_art, male_list, female_list, 100000))
    # print("get closest:", test_seq_embeddings.get_seq_closest('women', 1850))
    # print("get closest:", test_seq_embeddings.get_seq_closest('men', 1850))
    # print("get closet 1850-1860:", test_seq_embeddings.get_closet('men'))
    # print(test_seq_embeddings.get_closet('men').values())
    
    target_list_career = load_attrs("./WordList/career.txt")
    tarfet_list_family = load_attrs("./WordList/family.txt")
    
    print("SC_weat_ts_dict_career:", test_seq_embeddings.SC_WEAT_ts_dict(target_list_career, male_list, female_list))
    print("SC_weat_d_dict_career:", test_seq_embeddings.SC_WEAT_d_dict(target_list_career, male_list, female_list))
    print("SC_weat_p_dict_career:", test_seq_embeddings.SC_WEAT_p_dict(target_list_career, male_list, female_list, 100000))
    len(target_list_career)