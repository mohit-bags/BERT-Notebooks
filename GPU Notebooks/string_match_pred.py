'''
File: string_match_pred.py
Project: src
File Created: Saturday, 26th June 2021 03:38:47 am
Author: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Last Modified: Saturday, 26th June 2021 03:38:47 am
Modified By: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Copyright 2021 Shubham
'''


# importing libraries
import nltk
import pandas as pd
import numpy as np
import ast
import json
import re
# import swifter
from tqdm import tqdm
import time
from nltk.stem.wordnet import WordNetLemmatizer
# from PyDictionary import PyDictionary # for dictionary splitting
from collections import OrderedDict 


class BERT_prediction_single_summary():
    
    def __init__(self,input_from_bert_ip):
        
        self.summary = input_from_bert_ip
        if self.summary[0] == '[':
            res = ast.literal_eval(self.summary)
            self.summary = res
        else:
            x = []
            x.insert(0, self.summary)
            self.summary = x
        
        
        
    def read_file(self, tech_file_path):
        tech = pd.read_csv(tech_file_path)
        tech.columns = ["index", "title"]
        tech.drop(columns = ["index"], axis =1 , inplace = True)
        tech.reset_index(inplace = True, drop = True)
        self.tech = tech
        
        return tech
        
    def high_freq_eng_words(self, tech_file_path):
        
        r = pd.read_csv(tech_file_path)
        self.r = r
        for i in range(0, len(r)):
            r["Remove"][i] = r["Remove"][i].lower()
            remove_list = r["Remove"].tolist()
            remove_list.append("ve")
            self.remove_list = remove_list
            
        return remove_list
            
    def string_match(self):
        
        tech_list = []
        summary_list = []
        id_list = []
        
        self.tech['title'] = self.tech['title'].astype(str)
        all_tech_words = list(self.tech['title'].str.lower())
        
        tech_keys=[]
        tech_row=[]
        
        for k in all_tech_words:
            if k in self.summary[0] and len(k)>2:
                tech_row.append(k)
        
        def get_exact_match(txt, tech):
            try:
                patt = '|'.join(['\\b'+elem+'\\b' for elem in tech])
                matched_patt = re.findall(patt,txt)
            except Exception as e:
                print(e)
                matched_patt = []
            return matched_patt
        
        txt = self.summary[0]
        exact_matched_patt = get_exact_match(txt, tech_row)
        
        res = list(OrderedDict.fromkeys(exact_matched_patt))
        exact_matched_patt = res
        
        for word in list(exact_matched_patt):
            if word in self.remove_list:
                exact_matched_patt.remove(word)
                
        op = exact_matched_patt
        if len(op)==0:
            print("/n/n***No Match Found | You Gave an Empty Summary***\n\n")
        if len(op)!=0:
            print("\n\n*** Yes, We have a Match !! ***\n\n",op)
    
        return op
        





