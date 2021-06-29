'''
File: app_shubham_test.py
Project: src
File Created: Saturday, 25th June 2021 10:38:47 pm
Author: Mohit Bagaria
-----
Last Modified: Saturday, 26th June 2021 03:38:47 am
Modified By: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Copyright 2021 Shubham
'''


from bs4 import BeautifulSoup

import re
import nltk
import torch
import numpy as np
import pandas as pd
from simpletransformers.ner import NERModel,NERArgs
# import streamlit as st


class bert_pred:
    def __init__(self, 
                 model_path='/home/slintel/bert_folder/pkl_model/model_2000_non_multiprocessing', 
                 eng_data_path='/home/slintel/bert_folder/dataset/Eng_pydictionary_2.csv'):
        """
        """
        self.model_path = model_path
        self.eng_data_path = eng_data_path

        nltk.download("punkt")

        # Loading the model
        self._load_model()
    
    def _load_model(self):
        """
        """
        args = NERArgs()
#         args.num_train_epochs =2
#         args.learning_rate = 1e-4
#         args.overwrite_output_dir =True
#         args.train_batch_size = 32
#         args.eval_batch_size = 32
        args.use_multiprocessed_decoding = False
        self.model = NERModel('bert', 'bert-base-uncased',args = args, use_cuda = False)
#         self.model.args.use_multiprocessed_decoding = False

#         print("\n\n****** Model Loaded - Check 1 ******\n\n")

        self.eng_data = pd.read_csv(self.eng_data_path)
        self.eng_data = self.eng_data.drop(columns="Unnamed: 0")
        self.eng_data["English_word"] = self.eng_data["English_word"].apply(self._clean_text)

    def _clean_text(self, text):
        """
        """
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

        text = BeautifulSoup(text, "lxml").text # HTML decoding
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        return text

    def _get_sentence(self, sentence):
        """
        """    
#         print("\n\n****** get_sen() Run - Check 3 ******\n\n")
#         print("\n\n****** Model Run - Check 3.1 ******\n\n", self.model,"\n\n****** sentence Run - Check 3.2 ******\n\n",sentence)

        pred_data, _ = self.model.predict([sentence])

#         print("\n\n****** get_sen() Run-DONE - Check 4 ******\n\n")

        df = pd.DataFrame(pred_data)
        df = df.transpose()
        
        df.columns = ['AA']
        
        df['AA'] = df['AA'].astype(str)
        df['AA'] = df['AA'].str[2:-2]
        
        df[['Word', 'Pred']] = df['AA'].str.split("': '", 1, expand=True)
        
        df = df.drop(columns='AA')
        
        return df

    def _give_dataframe_return_list_of_tags(self, temp_data):
#         print("\n\n****** give_dataframe_return_list_of_TAGS() inside - Check 7 ******\n\n")
        tagg_per_sent = []
        prev_tag = "O"
        string = ""

        for i in range(0,len(temp_data)):
            value = temp_data['Pred'][i]
            key = temp_data['Word'][i]
            
            if(value == "B-ORG"):
                if (prev_tag == "B-ORG" or prev_tag == "I-ORG" or prev_tag == "O"):
                    if(len(string) > 0):
                        tagg_per_sent.append(string)
                    string = key
                prev_tag = "B-ORG"
                
            elif (value == "I-ORG"):
                string += " " + key
                prev_tag = "I-ORG"
            else:
                if(len(string) > 0):
                    tagg_per_sent.append(string)
                string = ""
                prev_tag = "O"
                
            if(len(string) > 0):
                tagg_per_sent.append(string)

        return tagg_per_sent

    def _get_final_output_predictions(self, bert_preds):
        """
        """
        eng_dict = dict()
        eng_list = list()
        output_predictions = list()


        for j in self.eng_data["English_word"]:
            if j not in eng_dict:
                eng_dict[j] = 1
            else:
                eng_dict[j] += 1
    
        for i in eng_dict:
            eng_list.append(i)

        # (all entries which are in eng dictionary)
        removed = 0
        kept = 0
        tags = bert_preds

        for j in tags:
            if j in eng_list:
                kept += 1
                output_predictions.append(j)
            else:
                removed += 1

        return output_predictions

    def predict(self, sentence_list):
        """
        """
        sentence = ''.join(sentence_list)

        # Step-1 Checking and cleaning the input sentence
        try:
            sentence = self._clean_text(text=sentence)
            print("\n\n****** Clean Text Run - Check 2 ******\n\n")
            sentence = ' '.join(sentence.split())
        except: 
            print("**Bad Input**")

        # Step-2 Calculate length of sentence
        len_of_sentence = len(sentence) #no of characters

        # Step-3 Predict Using Bert
        temp_data = self._get_sentence(sentence=sentence)
#         print("\n\n****** get_sen() Run return - Check 5 ******\n\n")
        len_till_now = temp_data['Word'].str.len().sum() + 1 + len(temp_data)

        # Step-4 Clipping the input
        while (len_till_now < len_of_sentence):
            temp_1 = self._get_sentence(sentence=sentence[len_till_now:])
            temp_data = temp_data.append(temp_1).reset_index(drop=True)
            len_till_now = temp_data['Word'].str.len().sum() + 1 + len(temp_data)
        
        # Step-5
#         print("\n\n****** _give_dataframe_return_list_of_tags() Run - Check 6 ******\n\n")

        bert_preds=list(set(self._give_dataframe_return_list_of_tags(temp_data=temp_data)))
#         print("\n\n****** give_dataframe_return_list_of_TAGS() return - Check 8 ******\n\n")

        # Step-6
        output_predictions = self._get_final_output_predictions(bert_preds=bert_preds)

        return output_predictions


# bert_obj = bert_pred(model_path='/home/slintel/bert_folder/pkl_model/model_2000_false_multiprocessed_decoding', 
#                      eng_data_path="/home/slintel/bert_folder/dataset/Eng_pydictionary_2.csv")

# bert_obj.predict(sentence_list=sentence_list)