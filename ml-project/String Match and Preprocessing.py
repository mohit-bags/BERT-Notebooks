#!/usr/bin/env python
# coding: utf-8


# libraries for preprocessing
import nltk
import pandas as pd
import numpy as np
import re

from collections import OrderedDict

backend = 'multiprocessing'


# In[42]:


def get_exact_match(txt, techt):
    try:
        patt = '|'.join(['\\b' + elem + '\\b' for elem in techt])
        matched_patt = re.findall(patt, txt)
    except Exception as e:
        print(e)
        matched_patt = []
    return matched_patt


# In[45]:


def save_file(df, name):
    df.to_csv(name, index=False)


# In[62]:


class preprocessing():

    def read_input_file(self, tech_file_path):
        data = pd.read_csv(tech_file_path)
        data.reset_index(inplace=True, drop=True)
        self.data = data
        return data

    def read_tech_list(self, tech_file_path):
        tech = pd.read_csv(tech_file_path)
        col_name = list(tech.columns)[-1]
        for i in range(len(tech[col_name])):
            tech[col_name][i] = self.clean_text(tech[col_name][i])
        return tech

    def read_remove_list(self, remove_file_path):
        r = pd.read_csv(remove_file_path)
        for i in range(0, len(r)):
            r["Remove"][i] = r["Remove"][i].lower()
            remove_list = r["Remove"].tolist()
        return remove_list

    def dump_format_change(self, df, col_name):
        for i in range(0, len(df)):
            print(i)
            if df[col_name][i] is not np.nan:
                demo = df[col_name][i]
                res = demo.strip('][')
                res = res.lower()
                res = res.replace(",", " , ")
                resl = [res]
                df[col_name][i] = resl

            else:
                df[col_name][i] = '[]'
                demo = df[col_name][i]
                res = demo.strip('][')
                resl = [res]
                df[col_name][i] = resl
        return df


    def start_cleaning(self, df, col_name):
        df = df.loc[:, df.notnull().any(axis=0)]
        df = self.dump_format_change(df, 'text_clean')
        df.reset_index(inplace=True, drop=True)
        return df


# In[67]:


class string_match():
    def create_tech_keys(self, tech, tech_col_name, df, df_col_name):
        tech['Tech'] = tech['Tech'].astype(str)
        all_tech_words = list(tech['Tech'].str.lower())

        eng['English_word'] = eng['English_word'].astype(str)
        all_eng_words = list(eng['English_word'].str.lower())

        summary["summaries_matching"] = summary["experience_summary"].astype(str)
        all_strings = list(summary["summaries_matching"].str.lower())

        i = 1
        j = 1
        tech_keys = []
        eng_keys = []
        for item in all_strings:
            #print("***", "TEST No tech,", i, "***")
            i = i + 1
            tech_row = []
            for k in all_tech_words:
                if k in item and len(k) > 2:
                    tech_row.append(k)
            tech_keys.append(tech_row)

            #print("***", "TEST No eng ,", j, "***")
            j = j + 1
            eng_row = []
            for k in all_eng_words:
                if k in item and len(k) > 2:
                    eng_row.append(k)
            eng_keys.append(eng_row)
        df['Tech_from_string_match'] = tech_keys
        df['Eng_from_string_match'] = eng_keys

    def start_string_match(self, tech, tech_col_name, df, df_col_name, remove_list):
        df = self.create_tech_keys(tech, tech_col_name, df, df_col_name)

        df['exact_matched_patt_eng'] = df.apply(
            lambda x: get_exact_match(x[df_col_name][0], x["Eng_from_string_match"]), axis=1)
        df['exact_matched_patt_tech'] = df.apply(
            lambda x: get_exact_match(x[df_col_name][0], x["Tech_from_string_match"]), axis=1)

        for ind in df.index:
            res = list(OrderedDict.fromkeys(df["exact_matched_patt_eng"][ind]))
            df["exact_matched_patt_eng"][ind] = res
        for ind in df.index:
            res = list(OrderedDict.fromkeys(df["exact_matched_patt_tech"][ind]))
            df["exact_matched_patt_tech"][ind] = res

        for i in range(0, len(df)):
            for word in list(df["exact_matched_patt_eng"][i]):
                if word in remove_list:
                    df["exact_matched_patt_eng"][i].remove(word)

        for i in range(0, len(df)):
            for word in list(df["exact_matched_patt_tech"][i]):
                if word in remove_list:
                    df["exact_matched_patt_tech"][i].remove(word)

        print("Creation Of Tags is Over. Please save the file via safe_file function.")
        return df


# In[72]:


summary_file_path = input()

# In[73]:


remove_list_path = input()

# In[74]:


print("Please input the file path to Tech dictionary")
tech_file_path = input()

print("Please input the file path to eng dictionary")
eng_file_path = input()

# In[58]:


x = preprocessing()

# In[48]:


remove_list = x.read_remove_list(remove_list_path)
summary = pd.read_csv(summary_file_path)
eng = pd.read_csv(eng_file_path)
tech = pd.read_csv(tech_file_path)

# In[59]:


tech = x.read_tech_list(tech_file_path)

# In[60]:


tech.head(20)

# In[61]:


test = x.start_cleaning(summary, 'summaries')

# In[68]:


y = string_match()

# In[69]:


test = y.start_string_match(tech, 'Tech_word', test, 'text_clean', remove_list)

# In[70]:


test['exact_matched_patt']

# In[71]:


save_file(test, 'final_run.csv')

# In[ ]:
