'''
File: app_shubham_test.py
Project: src
File Created: Saturday, 26th June 2021 03:38:47 am
Author: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Last Modified: Saturday, 26th June 2021 03:38:47 am
Modified By: Shubham Sunwalka (shubham.kumar@slintel.com>)
-----
Copyright 2021 Shubham
'''


from simpletransformers.ner import NERModel, NERArgs

import json 
import pandas as pd
import streamlit as st


st.title("Named Entity Recognition")

st.write("""
# Explore Us!
Which model is the best?
""")

model_name = st.sidebar.selectbox("Select Model",("bert","roberta"))
st.write(model_name)

sentence = st.text_input("Sentence")
st.write(sentence)

@st.cache(suppress_st_warning=True)
def load_model():
    args = NERArgs()
    args.use_multiprocessed_decoding = False
    model1 = NERModel('bert', 'bert-base-uncased', args=args, use_cuda=False)
    
    return model1


def predict(sentence):
    if sentence :
        # model1 = NERModel('bert', 'NERMODEL1',
        #           labels=["B-sector","I-sector","B-funda","O","operator","threshold","Join","B-attr","I-funda","TPQty","TPUnit","Sortby", 
        #                   "B-eco","I-eco","B-index","Capitalization","I-","funda","B-security",'I-security','Number','Sector','TPMonth','TPYr','TPRef'],
        #           args={"save_eval_checkpoints": False,
        # "save_steps": -1,
        # "output_dir": "NERMODEL",
        # 'overwrite_output_dir': True,
        # "save_model_every_epoch": False,
        # 'reprocess_input_data': True, 
        # "train_batch_size": 10,'num_train_epochs': 15,"max_seq_length": 64}, use_cuda=False)

        model1 = load_model()
        

        predictions, raw_outputs = model1.predict([sentence])     
        result = json.dumps(predictions[0])
        return result

if sentence :
    result= predict(sentence)
    #result=pd.DataFrame(result)
    st.write(result)












# """Streamlit v. 0.52 ships with a first version of a **file uploader** widget. You can find the
# **documentation**
# [here](https://streamlit.io/docs/api.html?highlight=file%20upload#streamlit.file_uploader).

# For reference I've implemented an example of file upload here. It's available in the gallery at
# [awesome-streamlit.org](https://awesome-streamlit.org).
# """
# from enum import Enum
# from io import BytesIO, StringIO
# from typing import Union

# import pandas as pd
# import streamlit as st

# STYLE = """
# <style>
# img {
#     max-width: 100%;
# }
# </style>
# """

# FILE_TYPES = ["csv", "py", "png", "jpg"]


# class FileType(Enum):
#     """Used to distinguish between file types"""

#     IMAGE = "Image"
#     CSV = "csv"
#     PYTHON = "Python"


# def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
#     """The file uploader widget does not provide information on the type of file uploaded so we have
#     to guess using rules or ML. See
#     [Issue 896](https://github.com/streamlit/streamlit/issues/896)

#     I've implemented rules for now :-)

#     Arguments:
#         file {Union[BytesIO, StringIO]} -- The file uploaded

#     Returns:
#         FileType -- A best guess of the file type
#     """

#     if isinstance(file, BytesIO):
#         return FileType.IMAGE
#     content = file.getvalue()
#     if (
#         content.startswith('"""')
#         or "import" in content
#         or "from " in content
#         or "def " in content
#         or "class " in content
#         or "print(" in content
#     ):
#         return FileType.PYTHON

#     return FileType.CSV


# def main():
#     """Run this function to display the Streamlit app"""
#     st.write("""
#         # Simple Iris Flower Prediction App
#         This app predicts the **Iris flower** type!
#         """)
#     #st.info(__doc__)
#     st.markdown(STYLE, unsafe_allow_html=True)

#     file = st.file_uploader("Upload file", type=FILE_TYPES)
#     show_file = st.empty()
#     if not file:
#         show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
#         return

#     file_type = get_file_type(file)
#     if file_type == FileType.IMAGE:
#         show_file.image(file)
#     elif file_type == FileType.PYTHON:
#         st.code(file.getvalue())
#     else:
#         data = pd.read_csv(file)
#         st.dataframe(data.head(10))

#     file.close()


# # main()