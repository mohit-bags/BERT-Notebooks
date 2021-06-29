#!/usr/bin/env python
# coding: utf-8

import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import torch

#         model = torch.load('/home/slintel/bert_folder/pkl_model/model_2000',map_location = 'cuda')


class bert_pred:
    
    eng_bert_tags=[]
    
    def __init__(self, sentence_list):
        
        sentence = ''.join(sentence_list)
        
        model = torch.load('/home/slintel/bert_folder/pkl_model/model_2000')
        print("\n\n****** Model Loaded - Check 1 ******\n\n")




        nltk.download("punkt")

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

        def clean_text(text):

            text = BeautifulSoup(text, "lxml").text # HTML decoding
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
            return text
        
        try:
            sentence=clean_text(sentence)
            print("\n\n****** Clean Text Run - Check 2 ******\n\n")
            sentence=' '.join(sentence.split())
        except: 
            print("**Bad Input**")



        def get_sen(sentence):
            
            print("\n\n****** get_sen() Run - Check 3 ******\n\n")
            pred_data,model_output=model.predict([sentence])
            print("\n\n****** get_sen() Run-DONE - Check 4 ******\n\n")
            df=pd.DataFrame(pred_data)
            df=df.transpose()
            df.columns=['AA']
            df['AA']=df['AA'].astype(str)
            df['AA']=df['AA'].str[2:-2]
            df[['Word', 'Pred']] = df['AA'].str.split("': '", 1, expand=True)
            df=df.drop(columns='AA')
            
            return df

        def give_dataframe_return_list_of_TAGS(temp_data):
          print("\n\n****** give_dataframe_return_list_of_TAGS() inside - Check 7 ******\n\n")
          tagg_per_sent = []
          prev_tag="O"
          string=""
          for i in range(0,len(temp_data)):
            value=temp_data['Pred'][i]
            key=temp_data['Word'][i]
            if(value=="B-ORG"):
              # print(key)
              if (prev_tag=="B-ORG" or prev_tag=="I-ORG" or prev_tag=="O"):
                    if(len(string)>0):
                      tagg_per_sent.append(string)
                    string=key
              prev_tag="B-ORG"

            elif (value=="I-ORG"):
              string+=" "+key
              prev_tag="I-ORG"

            else:
              if(len(string)>0):
                  tagg_per_sent.append(string)
              string=""
              prev_tag="O"
            if(len(string)>0):
                tagg_per_sent.append(string)

          return tagg_per_sent
#         print(sentence)
        length_shouldbe = len(sentence) #no of characters

        temp_data=get_sen(sentence)
        print("\n\n****** get_sen() Run return - Check 5 ******\n\n")
        len_till_now = temp_data['Word'].str.len().sum()+1+len(temp_data)

        while (len_till_now<length_shouldbe):
          aaa = get_sen(sentence[len_till_now:])
          kkk = temp_data.append(aaa)
          temp_data = kkk.reset_index(drop=True)
          len_till_now = temp_data['Word'].str.len().sum()+1+len(temp_data)
        
        print("\n\n****** give_dataframe_return_list_of_TAGS() Run - Check 6 ******\n\n")

        bert_preds=list(set(give_dataframe_return_list_of_TAGS(temp_data)))
        print("\n\n****** give_dataframe_return_list_of_TAGS() return - Check 8 ******\n\n")

        eng_data = pd.read_csv("/home/slintel/bert_folder/dataset/Eng_pydictionary_2.csv")
        eng_data=eng_data.drop(columns="Unnamed: 0")
        eng_data["English_word"]=eng_data["English_word"].apply(clean_text)
        eng_dict = {}
        for j in eng_data["English_word"]:
            if j not in eng_dict:
                eng_dict[j] = 1
            else:
                eng_dict[j] += 1
        eng_list=[]
        for i in eng_dict:
            eng_list.append(i)

        # (all entries which are in eng dictionary)
        bert_pred.eng_bert_tags=[]
        removed=0
        kept=0
        tags= bert_preds
        for j in tags:
            if j in eng_list:
                kept+=1
                bert_pred.eng_bert_tags.append(j)
            else:
                removed+=1
#         print(bert_preds)
#         print(eng_bert_tags)
#         eng_bert_per1sen=eng_bert_tags
    def get_eng_pred(self):
        return bert_pred.eng_bert_tags


# In[2]:


# inp = "marketing business administration and media industry professional with work experience in leading international companies startups and expertise in digital pr design creative materials account management sales licensing distribution and content acquisition fields sociable positive confident selfmotivated result driven tech savvy creative strong team player highly adaptable capable of multitasking passionate about my job intuitive leadership skills execute tasks swiftly and effectively entrepreneurial ability and experience good analytical skills and knowledge international mindset open to new possibilities and new industries moving abroad and extensive business travels currently living in haifa israel computer skills ms office sap adobe cs apple iwork imovie final cut pro various digital marketing tools marketing business development and media industry professional with work experience in leading international high tech companies startups and expertise in project management performance marketing account management sales digital pr brand management creative materials design licensing distribution and content acquisition fields sociable positive confident selfmotivated result driven tech savvy creative strong team player highly adaptable capable of multitasking passionate about my job intuitive leadership skills execute tasks swiftly and effectively entrepreneurial ability and experience good analytical skills and knowledge international mindset open to new possibilities and new industries moving abroad and extensive business travels based in haifa israel but open to relocation for a job marketing and entertainment industry professional with work experience in leading international film and media companies and expertise in marketing design creative materials digital marketing pr account management sales licensing distribution and content acquisition fields sociable positive confident selfmotivated result driven tech savvy creative strong team player highly adaptable capable of multitasking passionate about my job intuitive leadership skills execute tasks swiftly and effectively entrepreneurial ability and experience good analytical skills and knowledge international mindset open to new possibilities and new industries moving abroad and extensive business travelsnncomputer skills ms office sap adobe cs apple iwork imovie final cut pro various digital marketing tools marketing business development and media industry professional with work experience in leading international companies startups and expertise in digital pr brand management design creative materials account management sales licensing distribution and content acquisition fields sociable positive confident selfmotivated result driven tech savvy creative strong team player highly adaptable capable of multitasking passionate about my job intuitive leadership skills execute tasks swiftly and effectively entrepreneurial ability and experience good analytical skills and knowledge international mindset open to new possibilities and new industries moving abroad and extensive business travels recently moved to haifa israel but open to relocation for a job marketing and pr professional with over 6 years of experience looking for opportunities in israel marketing project management professional with over 8 years of experience promotions marketing manager at nbcuniversal inc"
# inp = input()['full stack developer 9+ years of experience leading small (5-10) development teams in large-scale projects. proficient in java & javascript; backend frameworks such as spring, struts, hibernate; frontend frameworks angular js, protractor, jasmine. db mainly oracle but also mysql, sqlserver; wide variety of bussiness domains working for companies like cars.com, credit suisse, qualcomm, hewlett-packard, inter-american development bank, bank of america & monsanto. experience in agile methodologies kanban & scrum; test driven development. always looking for new challenges and learn new technologies., architect at globant - credit suisse, tech lead at globant - cars.com, scjp 6 certified programmer, tech lead at softtek']
# inp = input()

# In[3]:


# preds = bert_pred(inp)
# eng_preds=preds.get_eng_pred()


# In[4]:


# print(eng_preds)


# In[ ]:




