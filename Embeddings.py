#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:37:53 2024

@author: tuituiwang
"""

import heapq

import numpy as np
from sklearn import preprocessing

# import IOUtils
# dir(IOUtils) 
import ioutils
# dir(ioutils)

# from ioutils import load_pickle


import collections
import random
import math
#, lines

class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        '''
        Parameters
        ----------
        vecs : A NumPy array 
            where each row corresponds to the vector representation of a word.
        vocab : A list of words (vocabulary)
            each word corresponds to a row in vecs.
        normalize :  A boolean
            ndicating whether to normalize the vectors upon initialization.
        **kwargs : Additional keyword arguments.

        Functionality:
        -------
        Sets up the word vectors (self.m), dimension of vectors (self.dim), and vocabulary.
        Creates a mapping from words to indices (self.wi) and optionally normalizes the vectors.

        '''
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        '''
        Purpose
        -------
        Allows indexing into the Embedding object to get the vector representation of a word.

        Parameters
        ----------
        key 

        Raises
        ------
        KeyError
            if the word is out-of-vocabulary (OOV)

        Returns
        -------
        the vector representation of the word.
        '''
        
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        '''
        Purpose: Makes the Embedding object iterable over the vocabulary.

        Returns
        -------
        Iterates over self.iw, which is the vocabulary list.

        '''
        return self.iw.__iter__()

    def __contains__(self, key):
        '''
        Purpose: Checks if a word is in the vocabulary.

        Parameters
        ----------
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        Returns True if the word is in the vocabulary, False otherwise.

        '''
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=False, **kwargs):
        '''
        

        Parameters
        ----------
        path : string
            Path to the files containing the embeddings and vocabulary..
        normalize : Boolean, optional
            Whether to normalize the embeddings after loading. 
            The default is True.
        add_context : Boolean, optional
            Whether to add context vectors (used for models with context information). 
            The default is False.
        **kwargs

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        mat = np.load(path + "-w.npy", mmap_mode="c")
        if add_context:
            mat += np.load(path + "-c.npy", mmap_mode="c")
        iw = ioutils.load_pickle(path + "-vocab.pkl")
        return cls(mat, iw, normalize) 

    def get_subembed(self, word_list, **kwargs):
        '''
        Purpose: Filters out words that are out-of-vocabulary.

        Parameters
        ----------
        word_list : List of words
            keep in the new embedding.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            a new Embedding object with the vectors for the selected words.

        '''
        word_list = [word for word in word_list if not self.oov(word)]
        keep_indices = [self.wi[word] for word in word_list]
        return Embedding(self.m[keep_indices, :], word_list, normalize=False)

    def reindex(self, word_list, **kwargs):
        '''
        Purpose
        -------
        Creates a new matrix of vectors for the provided word list, 
        filling in zeros for OOV words.

        Parameters
        ----------
        word_list : List of words 
        to include in the reindexed embedding.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        Returns a new Embedding object with the reindexed vectors.

        '''
        new_mat = np.empty((len(word_list), self.m.shape[1]))
        valid_words = set(self.iw)
        for i, word in enumerate(word_list):
            if word in valid_words:
                new_mat[i, :] = self.represent(word)
            else:
                new_mat[i, :] = 0 
        return Embedding(new_mat, word_list, normalize=False)

    def get_neighbourhood_embed(self, w, n=1000):
        '''
        Purpose
        -------
        Retrieves embeddings for the nearest neighbors of a given word.
	•	Parameters:
	•	
	•	Functionality:
	•	

        Parameters
        ----------
        w: The target word to find neighbors for.
    	n: Number of nearest neighbors to retrieve.

        Returns
        -------
        Finds the nearest neighbors of w and creates a new Embedding object with their vectors.

        '''
        neighbours = self.closest(w, n=n)
        keep_indices = [self.wi[neighbour] for _, neighbour in neighbours] 
        new_mat = self.m[keep_indices, :]
        return Embedding(new_mat, [neighbour for _, neighbour in neighbours]) 

    def normalize(self):
        '''
        Normalize the vectors

        '''
        preprocessing.normalize(self.m, copy=False)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w):
        '''
        Parameters
        ----------
        w : TYPE
            The word to retrieve the vector for.

        Returns
        -------
        TYPE
        the vector for the word if it is in the vocabulary; 
        otherwise, returns a zero vector and prints an OOV message.
        '''
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print("OOV: ", w)
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        Parameters
        ---------
		w: The target word.
		n: Number of closest neighbors to find.
        Returns
        -------
        Computes the similarity scores of all words to the target word
        returns the top n closest words
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))
    
    #WEAT test
    def weat_rand_test(self, target_list1, target_list2, attr_list1, attr_list2, iterations):
        u_words = target_list1 + target_list2
        runs = np.min((iterations, math.factorial(len(u_words))))
        
    
        original = self.test_statistic(target_list1, target_list2, attr_list1, attr_list2)
        seen = set()
        r = 0
        for _ in range(runs):
            permutation = tuple(random.sample(u_words, len(u_words)))
            if permutation not in seen:
                hat_1 = permutation[0:len(target_list1)]
                hat_2 = permutation[len(target_list2):]
                if self.test_statistic(hat_1, hat_2, attr_list1, attr_list2) > original:
                    r += 1
                seen.add(permutation)
        p_value = r / runs
        return p_value
    
    def cohen_d(self, target_list1, target_list2, attr_list1, attr_list2, iterations):
        if len(target_list1) == 0 or len(target_list2) == 0:
            return "NA"
        sum_1, sum_2 = self.test_sum(target_list1, target_list2, attr_list1, attr_list2)
        mean_1 = sum_1 / len(target_list1)
        mean_2 = sum_2 / len(target_list2)
        m_u_f = np.array([self.get_mean_diff(w, attr_list1, attr_list2) for w in target_list1 + target_list2])
        stdev = m_u_f.std(ddof=1)
        return (mean_1 - mean_2) / stdev
    
    def test_statistic(self, target_list1, target_list2, attr_list1, attr_list2):
        sum_1, sum_2 = self.test_sum(target_list1, target_list2, attr_list1, attr_list2)
        return sum_1 - sum_2
    
    
    def test_sum(self, target_list1, target_list2, attr_list1, attr_list2):
        sum_1 = 0.0
        sum_2 = 0.0
        for t in target_list1:
            sum_1 += self.get_mean_diff(t, attr_list1, attr_list2)
        for t in target_list2:
            sum_2 += self.get_mean_diff(t, attr_list1, attr_list2)
        return sum_1, sum_2
    
    def get_mean_diff(self, w, list_1, list_2):
        mean_1 = np.mean([self.similarity(w, word) for word in list_1])
        mean_2 = np.mean([self.similarity(w, word) for word in list_2])
        # print(mean_1 - mean_2)
        return mean_1 - mean_2
    
    #SC_WEAT
    def SC_WEAT_test(self, target_list, attr_list1, attr_list2, iterations = 10000):
        u_words = attr_list1 + attr_list2
        runs = np.min((iterations, math.factorial(len(u_words))))
        original = self.SC_WEAT_ts(target_list, attr_list1, attr_list2)
        seen = set()
        count = 0
        # count_greater = 0
        # count_less = 0
        
        for _ in range(runs):
            permutation = tuple(random.sample(u_words, len(u_words)))
            if permutation not in seen:
                hat_1 = permutation[:len(attr_list1)]
                hat_2 = permutation[len(attr_list1):]
                
                permuted_stat = self.SC_WEAT_ts(target_list, hat_1, hat_2)
                
                if abs(permuted_stat) > abs(original):
                    count += 1
        #         if permuted_stat > original:
        #             count_greater += 1
        #         elif permuted_stat < original:
        #             count_less += 1
                
                seen.add(permutation)
        
        # # Compute two-sided p-value
        # p_value_greater = (count_greater + 1) / (runs + 1)  # Adding 1 to avoid division by zero
        # p_value_less = (count_less + 1) / (runs + 1)       # Adding 1 to avoid division by zero
        # two_sided_p_value = min(p_value_greater, p_value_less) * 2  # Two-sided p-value
        two_sided_p_value = (count + 1) / (runs + 1)
        
        return two_sided_p_value
    
    def SC_WEAT_d(self, target_list, attr_list1, attr_list2):
        if len(target_list) == 0:
            return "NA"

        # Calculate the SC-WEAT test statistic for the observed data
        observed_statistic = self.SC_WEAT_ts(target_list, attr_list1, attr_list2)
        
        # Compute the mean and standard deviation of mean differences
        mean_diffs = np.array([self.get_mean_diff(w, attr_list1, attr_list2) for w in target_list])
        stdev = mean_diffs.std(ddof=1)
        
        if stdev == 0:
            return "NA"  # Avoid division by zero
        
        # Cohen's d
        cohen_d = observed_statistic / stdev
        return cohen_d
            
    
    #test statistics
    # def SC_WEAT_ts_2(self, target_list, attr_list1, attr_list2):
    #     SC_WEAT_ts = 0.0
    #     for w in target_list:
    #         SC_WEAT_ts += self.get_mean_diff(w, attr_list1, attr_list2)
    #     return SC_WEAT_ts / len(target_list)
        # return SC_WEAT_ts
    
    def SC_WEAT_ts(self, target_list, attr_list1, attr_list2):
        # w1_score = 0
        # w2_score = 0
        w_diff = []
        for w in target_list:
            # w1_score = np.mean([self.similarity(w, w1) for w1 in attr_list1])
            # w2_score = np.mean([self.similarity(w, w2) for w2 in attr_list2])
            w_mean_diff = self.get_mean_diff(w, attr_list1, attr_list2)
            # w_diff.append(w1_score - w2_score)
            w_diff.append(w_mean_diff)
            # w_diff += w1_score - w2_score
        
        return np.mean(w_diff)
        
        
        
        


if __name__ == "__main__":
    def load_attrs(path):
        text = open(path, "r")
        text_list = list(set(text.read().lower().splitlines()))
        return text_list
    target_list_art = load_attrs("./WordList/arts.txt")
    target_list_math = load_attrs("./WordList/math.txt")
    female_list = load_attrs("./WordList/female_terms.txt")
    male_list = load_attrs("./WordList/male_terms.txt")
    target_list_career = load_attrs("./WordList/career.txt")
    tarfet_list_family = load_attrs("./WordList/family.txt")
    test_embeddings = Embedding.load("/Users/tuituiwang/Documents/PG/diss/Methodology/Data/COHA_word_sgns/1900")
    # print(test_embeddings.closest("men", 10))
    # print("mean_diff", test_embeddings.get_mean_diff(target_list_art[1], male_list, female_list))
    # print("test sums:", test_embeddings.test_sum(target_list_math, target_list_art, male_list, female_list))
    #print("cohen d:", test_embeddings.cohen_d(target_list_math, target_list_art, male_list, female_list, 100000))
    #print("p_value:", test_embeddings.weat_rand_test(target_list_math, target_list_art, male_list, female_list, 100000))
    print('art_SC_ts2:', test_embeddings.SC_WEAT_ts_2(target_list_art, male_list, female_list))
    print('art_SC_ts_2:', test_embeddings.SC_WEAT_ts(target_list_art, male_list, female_list))
    # print('art_SC_d:', test_embeddings.SC_WEAT_d(target_list_art, male_list, female_list))
    # print('art_SC_p:', test_embeddings.SC_WEAT_test(target_list_art, male_list, female_list, 100000))
    # print('career_ts_2:', test_embeddings.SC_WEAT_ts(target_list_career, male_list, female_list))
    # print('career_SC_d:', test_embeddings.SC_WEAT_d(target_list_career, male_list, female_list))
    # print('career_SC_p:', test_embeddings.SC_WEAT_test(target_list_career, male_list, female_list, 100000))
    # print('math_SC_d:', test_embeddings.SC_WEAT_test(target_list_math, male_list, female_list))
    # print(target_list_art)
