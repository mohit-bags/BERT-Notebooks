import nltk
import pandas as pd
import numpy as np
import ast
import json
import re
from bs4 import BeautifulSoup
import torch

class bert_pred:
  '''
  ___init__ : Provide -  Model Path 
              User Input for Prediction in List Format
              User Input is cleaned
  read_eng_list: INPUT: English Dictionary List path 
                 RETURNS: Cleaned English Dictionary
  clean_text : MODIFY this for any changes, USED REGEX and nltk
              INPUT: a string 
              RETURNS: the cleaned string

  complete_pred: Prediction is made by calling this FUNCTION
                INPUT: None
                RETURN: List of BERT Tags for the given User Input

  remove_non_eng_preds: INPUT: Cleaned English Dictionary and List of BERT Tags
                        RETURNS : List of BERT Tags containing words from ENGLISH DICTIONARY only

  get_sen: ****DONT MODIFY**** -> in-built function to generate predictions from the BERT Model
           INPUT: String
           RETURNS: DataFrame of 2 columns of Words with it's Tags
  
  give_dataframe_return_list_of_TAGS: ****DONT MODIFY**** -> in-built function to extract Relevant Tags
           INPUT: DataFrame returned from get_sen function
           RETURNS: List of BERT Tags

  '''
  def __init__(self,model_path):    #GIVE model path
    print("Give sentence as a list:")
    self.sentence_list = input() #Taking input as a list
    self.sentence = ''.join(self.sentence_list)
    self.sentence = self.clean_text(self.sentence)
    self.model= torch.load(model_path)
  
  def read_eng_list(self, eng_file_path): #give Eng dictionary path
    eng = pd.read_csv(eng_file_path)
    col_name = list(eng.columns)[-1]
    eng[col_name]=eng[col_name].apply(self.clean_text)
    return eng  #returning Cleaned english dictionary

  def clean_text(self, text):
    text = text.lower()
    text = re.sub(r'^\s*$', " ",text)
    text = re.sub('\s+', " ", text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = re.sub('\s+',' ',text)
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text   

  def get_sen(self,sentence): #generating bert preds and converting it to dataframe
    pred_data,model_output=self.model.predict([sentence])
    df=pd.DataFrame(pred_data)
    df=df.transpose()
    df.columns=['temp_col']
    df['temp_col']=df['temp_col'].astype(str)
    df['temp_col']=df['temp_col'].str[2:-2]
    df[['Word', 'Pred']] = df['temp_col'].str.split("': '", 1, expand=True)
    df=df.drop(columns='temp_col')

    return df #returning dataframe

  def give_dataframe_return_list_of_TAGS(self,temp_data): #givig complete BERT predicted Dataframe and collectin tags
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
    print("BERT Prediction completed")
    return tagg_per_sent #returning BERT predictions

  def complete_pred(self):
    length_shouldbe = len(self.sentence) #no of characters
    temp_data=self.get_sen(self.sentence) 
    len_till_now = temp_data['Word'].str.len().sum()+1+len(temp_data)

    while (len_till_now<length_shouldbe):
      aaa = self.get_sen(self.sentence[len_till_now:])
      kkk = temp_data.append(aaa)
      temp_data = kkk.reset_index(drop=True)
      len_till_now = temp_data['Word'].str.len().sum()+1+len(temp_data)

    bert_preds=list(set(self.give_dataframe_return_list_of_TAGS(temp_data)))

    return bert_preds
  
  def remove_non_eng_preds(self,eng_data,bert_preds):
    col_name=list(eng_data.columns)[-1]   
    eng_list = list(set(eng_data[col_name].tolist()))

    # (all entries which are in eng dictionary)
    eng_bert_tags=[]
    removed=0
    kept=0
    tags=bert_preds
    for j in tags:
        if j in eng_list:
            kept+=1
            eng_bert_tags.append(j)
        else:
            removed+=1
    print("Non-Eng Labels removed")
    return eng_bert_tags

print(bert_pred.__doc__) #DOCSTRING

model_path="/content/drive/MyDrive/model_2000_manually_ROW_WISE"
preds = bert_pred(model_path) #object of the class creted here

eng_data_path = "/content/drive/MyDrive//Eng_pydictionary_2.csv"
cleaned_eng_data = preds.read_eng_list(eng_data_path) #cleaning of eng dictionary

bert_preds = preds.complete_pred()
print(bert_preds)

eng_bert_preds = preds.remove_non_eng_preds(cleaned_eng_data,bert_preds)
print(eng_bert_preds)

