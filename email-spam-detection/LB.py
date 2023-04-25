import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from nltk.stem.porter import PorterStemmer

import psycopg2

#initialize connection
# @st.experimental_singleton
# def init_connection():
#     return psycopg2.connect(**st.secrets["postgres"])
#
# conn=init_connection()
# print('connection established')
#
# #perform query
# @st.experimental_memo(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         conn.commit()
#         count=cur.rowcount
#         print(count,'record inserted')

ps = PorterStemmer()


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)
def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

vectorizer = pickle.load(open('vectorizer_1.pkl','rb'))
model = pickle.load(open('model_1.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_preprocess(input_sms)
    transform_sms=stemmer(transformed_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
        # query=f'''
        # insert into "Dataset" (target,text)
        # values('spam','{input_sms}');
        # '''
        # run_query(query)
        # print('query executed')
    else:
        st.header("Not Spam")
        # query = f'''
        #         insert into "Dataset" (target,text)
        #         values('ham','{input_sms}');
        #         '''
        # run_query(query)
        # print('query executed')