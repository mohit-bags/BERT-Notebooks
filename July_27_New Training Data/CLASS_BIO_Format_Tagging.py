import numpy as np
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

class BIO_Tagger():
    '''
__init__ : INPUT- Mandatory - Data loading, and column definitions

clean_text : changes can be made as per your requirement
             INPUT- text
             RETURNS- cleaned text 

give_tags: INPUT- List of Tags
           RETURNS- Two lists (single word Tag, Multi-word Tag)

find_all : To Get all occurences(index) of a string

convert : Spliting mutli-words by space and returning them as a list

BIO_conversion : INPUT- sentence 
                 RETURNS- dataframe(with words of a sentence in column) with its TAG
                 LOGIC - First get Single and Multi-words tags,
                         Search for all occurences of the Multi-words tags 
                         and storing those indexes to Tag Multi-words when encountered handled those indexes
                         seperately, cause only they will be tagged as 'I'
runner: INPUT- None
        USE - 1. Cleans the sentence(Calls clean_text function)
              2. Converts manual tags into iterable list format
              3. Calls BIO_conversion for each sentence in the data
        RETURNS: DataFrame with BIO Tagging
    '''
    def __init__(self,data_path,manual_tag_col,text_col):
        self.data= pd.read_csv(data_path) #DATA PATH
        self.manual_tag_col = manual_tag_col #ENTER MANUAL TAGS COLUMN NAME
        self.text_col = text_col  #ENTER COLUMN NAME of sentence
   
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        text = BeautifulSoup(text, "lxml").text # HTML decoding
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        return text
    
    def give_tags(self,manual_tags): #give a list of strings then returns all tags as single & multi words
        tech_dict = {}
        for j in manual_tags:
            if j not in tech_dict:
                tech_dict[j] = 1
            else:
                tech_dict[j] += 1
        tech_list=[]
        tech_multi_words=[]
        for i in tech_dict:
            if ((' ' in i) == True):
                tech_multi_words.append(i)
            else:
                tech_list.append(i)
        return tech_list,tech_multi_words
    
    def find_all(self,a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub) # use start += 1 to find overlapping matches
        
    def convert(self,lst):
        #break words
        return ' '.join(lst).split()
    
    def BIO_conversion(self,sentence,manual_tags):
       
        tech_list,tech_multi_words = self.give_tags(manual_tags)
        ind_dict = {}
        for i in tech_multi_words:
            if i in sentence:
                #ind_dict[sentence.index(i)]=i #
                all_occ = list(self.find_all(sentence,i)) # [0, 5, 10, 15]
                for ind in all_occ:
                    ind_dict[ind]=i #word as value, key as index

        temp_word=""
        final_word_sen=[]
        final_tag_sen = []
        start=-1
        end=-1
        for i in range(0, len(sentence)):
            if(i in range(start,end)): #if the index right now is within a multi word TAG, we dont need to check it
                continue
            if sentence[i].isspace()==True and len(temp_word)>0: #on encountering a SPACE push to word
                final_word_sen.append(temp_word)
                if temp_word in tech_list: #checking if word is in a single word TAG
                    final_tag_sen.append("B")
                else:                      #since word is not a Single word TAG, give it O (OUSTIDE) tag
                    final_tag_sen.append("O") 
                temp_word="" #resetting the word
            else: #not space
                try: #do we have multiword at this index
                    temp_word=ind_dict[i] #GET THE MULTI WORD TAG
                    start=i+1              #storing the starting index
                    end=i+len(temp_word) #here tag these multiwords and update new i
                    listofwords = self.convert([temp_word]) #get those words splitted
                    f=True
                    for word in listofwords:
                        final_word_sen.append(word) #pushing the word
                        if f:
                            final_tag_sen.append("B") #Starting of Multi WORD (appending the TAG)
                            f=False                   #MARK FLAG as false, rest of the multi words are I(INSIDE TAG)
                        else: 
                            final_tag_sen.append("I") # appending the TAG
                    temp_word=""
                except: #not a space and not a multiword TAG, so just concatenate the string
                    if(sentence[i]!=' '):
                        temp_word+=sentence[i]
        return pd.DataFrame(list(zip(final_word_sen, final_tag_sen)),columns =['Word', 'Tag'])
    
    def runner(self):
        self.data[self.manual_tag_col] = [ [] if x is np.NaN else x for x in self.data[self.manual_tag_col] ]
        self.data[self.manual_tag_col]=self.data[self.manual_tag_col].astype(str)
        self.data[self.manual_tag_col] = self.data[self.manual_tag_col].apply(eval) #to convert string to list of strings

        self.data[self.text_col]=self.data[self.text_col].apply(self.clean_text)
        self.data[self.text_col] = self.data[self.text_col].replace('\s+', ' ', regex=True)
        final_data=pd.DataFrame()
        for i in range(0,len(self.data)):
            temp = self.BIO_conversion(self.data[self.text_col][i],self.data[self.manual_tag_col][i])
            length = len(temp)
            word="Sentence :"+str(i+1) #sentence no.
            a=[word]*length
            temp.insert(0,"Sentence #",a)
            final_data = final_data.append(temp, ignore_index=True) #appending sentences in the required format
        final_data.loc[(final_data['Tag'] == 'B'), 'Tag'] = 'B-ORG'
        final_data.loc[(final_data['Tag'] == 'I'), 'Tag'] = 'I-ORG'
        return final_data
        