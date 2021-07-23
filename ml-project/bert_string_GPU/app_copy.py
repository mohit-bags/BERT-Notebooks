'''
File: app.py
Project: src
File Created: Saturday, 26th June 2021 03:38:47 am
Author: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Last Modified: Saturday, 26th June 2021 03:38:47 am
Modified By: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Copyright 2021 Shubham
'''




import streamlit as st
from string_match_pred import BERT_prediction_single_summary as sg
import pickle
import torch
# from bert_inference_shubham import bert_pred as bt
from bert_inference_shubham_copy import bert_pred as bt



@st.cache(allow_output_mutation=True)
def get_string_match_prediction(inp):
    
    sm = sg(inp)
    tech = sm.read_file("../dataset/Tech_pydictionary_2.csv")
    remove_list = sm.high_freq_eng_words("../dataset/Remove.csv")
    string_match_op = sm.string_match()
    
    return string_match_op

@st.cache(allow_output_mutation=True)
def get_bert_prediction():
    bert_obj = bt()
    return bert_obj

# @st.cache(allow_output_mutation=True)
# def get_bert_string_match_prediction(inp):
    
#     sm = sg(inp)
#     tech = sm.read_file("../dataset/Tech_pydictionary_2.csv")
#     remove_list = sm.high_freq_eng_words("../dataset/Remove.csv")
#     string_match_op = sm.string_match()

#     preds = bt(inp)
#     bert_op=preds.get_eng_pred()
    
#     return bert_op+string_match_op


  

# this is the main function in which we define our webpage  
def main():
    
    
    st.title("Technology/Company Extraction from Linkedin Summary by Slintel")
    st.subheader("NER + String Match")
    st.markdown("""
    	#### Description
    	+ Entity Extraction App Demo
    	""")
    
    bert_obj = get_bert_prediction()

    # Named Entity Extraction
    
    if st.checkbox("String Match Tags"):
        st.subheader("Extract From Text")
        message = st.text_area("Enter Text","")
        if st.button("Extract", key="a1"):
            entity_result = get_string_match_prediction(message)
            st.json(entity_result)
            #st.success(entity_result)

    if st.checkbox("BERT Tags"):
        st.subheader("Extract From Text")
        message1 = st.text_area("Enter Text","",key="abc456")
        if st.button("Extract",key="a2"):
            entity_result1 = bert_obj.predict(message1)
            st.json(entity_result1)
            #st.success(entity_result)
            
    if st.checkbox("Extract All Tags"):
        st.subheader("Extract From Text")
        message2 = st.text_area("Enter Text","",key="abc123")
        if st.button("Extract",key="a3"):
            result_string = get_string_match_prediction(message2)
            result_bert = bert_obj.predict(message2)
            entity_result2 = result_string + result_bert
            st.json(entity_result2)
            #st.success(entity_result)
            

            
            

    st.sidebar.subheader("About App")
    st.sidebar.text("Slintel")
    st.sidebar.info("This is a confidential demo")
    
    st.sidebar.subheader("By")
    st.sidebar.text("Slintel Data Science Team")    
    
    
if __name__=='__main__': 
    main()







# # Core Pkgs
# import streamlit as st 
# import os


# # NLP Pkgs
# # from textblob import TextBlob 
# import spacy

# @st.cache
# def entity_analyzer(my_text):
# 	nlp = spacy.load('en_core_web_sm')
# 	docx = nlp(my_text)
# 	tokens = [ token.text for token in docx]
# 	entities = [(entity.text,entity.label_)for entity in docx.ents]
# 	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
# 	return allData


# def main():
# 	""" NLP Based App with Streamlit """

# 	# Title
# 	st.title("NLPiffy with Streamlit")
# 	st.subheader("Natural Language Processing On the Go..")
# 	st.markdown("""
#     	#### Description
#     	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
#     	Tokenization,NER,Sentiment,Summarization
#     	""")

# 	# Entity Extraction
# 	if st.checkbox("Show Named Entities"):
# 		st.subheader("Analyze Your Text")

# 		message = st.text_area("Enter Text","Type Here ..")
# 		if st.button("Extract"):
# 			entity_result = entity_analyzer(message)
# 			st.json(entity_result)



# 	st.sidebar.subheader("About App")
# 	st.sidebar.text("NLPiffy App with Streamlit")
# 	st.sidebar.info("Cudos to the Streamlit Team")
	

# 	st.sidebar.subheader("By")
# 	st.sidebar.text("Jesse E.Agbe(JCharis)")
# 	st.sidebar.text("Jesus saves@JCharisTech")
	

# if __name__ == '__main__':
# 	main()