#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:42:15 2024

@author: tuituiwang
"""

# import latexify
import numpy as np
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ast
import sys
# import pylab
# from utilities import *
# from statsmodels.formula.api import ols
# from statsmodels.iolib.summary2 import summary_col
# import statsmodels.api as sm
from scipy.stats import kendalltau
import copy
import scipy
from scipy.stats import pearsonr
# from  more_itertools import unique_everseen
import os

sns.set(style="whitegrid")

def heatmap_pearson_corr(difs_folder, plotsfolder, saveformat  = 'png'):
    for f in os.listdir(difs_folder):
        print(f)
        # Load the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(difs_folder, f), index_col=False)
    
        # Initialize heatmap arrays
        num_columns = len(df.columns[1:])# Adjust if needed
        print('columns number', num_columns)
        heatmap = np.zeros((num_columns, num_columns))
        heatmap_pvalues = np.zeros((num_columns, num_columns))
        
        
        # Iterate over columns starting from index 1
        for year1 in range(1, num_columns):
            print("year1", year1)
            if year1 == 11:
                print('year 1 when 11', df.iloc[:, year1])
            for year2 in range(1, num_columns):
                # Print information for debugging
                print("year2", year2)
                if year2 == 11:
                    print('year 2 when 11', df.iloc[:, year2])
                
                heatmap[year1, year2], heatmap_pvalues[year1, year2]  = pearsonr(df.iloc[:, year1], df.iloc[:, year2])
        axx = sns.heatmap(heatmap, annot = True, fmt = ".2f", xticklabels = df.columns[1:], yticklabels = df.columns[1:], 
                          robust = True, cbar = False, cmap = 'YlGnBu', annot_kws={"color": 'black', 'fontsize' : 8})
        # plt.xlabel('Year')
        # plt.ylabel('Year')
        for labelll in axx.get_yticklabels():
            labelll.set_size(11)
            labelll.set_weight("bold")
        for labelll in axx.get_xticklabels():
            labelll.set_size(11)
            labelll.set_weight("bold")
        plt.yticks(rotation=0)
        plt.tight_layout()
        if not os.path.exists(plotsfolder):
            os.makedirs(plotsfolder)
        plt.savefig(plotsfolder + 'correlationheatmap_distancestoself{}.{}'.format(
            os.path.splitext(f)[0], saveformat), dpi = 1000)
        plt.close()

heatmap_pearson_corr('./dif_by_words', './heatmaps/')

'''
Testing

df = pd.read_csv('./dif_by_words/B4appearance.csv', index_col=False)
num_columns = len(df.columns[1:])# Adjust if needed
print('columns number', num_columns)
heatmap = np.zeros((num_columns, num_columns))
heatmap_pvalues = np.zeros((num_columns, num_columns))


# Iterate over columns starting from index 1
for year1 in range(1, num_columns):
    print("year1", year1)
    if year1 == 11:
        print('year 1 when 11', df.iloc[:, year1])
    for year2 in range(1, num_columns):
        # Print information for debugging
        print("year2", year2)
        if year2 == 11:
            print('year 2 when 11', df.iloc[:, year2])
        
        heatmap[year1, year2], heatmap_pvalues[year1, year2]  = pearsonr(df.iloc[:, year1], df.iloc[:, year2])

axx = sns.heatmap(heatmap, annot = True, fmt = ".2f", xticklabels = df.columns[1:], yticklabels = df.columns[1:], 
                  robust = True, cbar = False, cmap = 'YlGnBu', annot_kws={"color": 'black', 'fontsize' : 8})
# plt.xlabel('Year')
# plt.ylabel('Year')
for labelll in axx.get_yticklabels():
    labelll.set_size(11)
    labelll.set_weight("bold")
for labelll in axx.get_xticklabels():
    labelll.set_size(11)
    labelll.set_weight("bold")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('heatmap_corr.png', dpi = 1000)
plt.close()

'''



























    