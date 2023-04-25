import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from PIL import Image
import warnings
import psycopg2

warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=UserWarning)
# Initialize connection
@st.cache_resource
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()
print('Connection Established')

#perform Query
@ st.cache_data(ttl=200)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
        count = cur.rowcount
        print(count,"Record Inserted")
        return cur.execute(query)

def spam_data(input_sms):
    query = f'''
                    insert into "Dataset" (target,text)
                    values('spam','{input_sms}');
                    '''
    run_query(query)
    print('Query Executed')

def ham_data(input_sms):
    query = f'''
                    insert into "Dataset" (target,text)
                    values('ham','{input_sms}');
                    '''
    run_query(query)
    print('Query Executed')

ps = PorterStemmer()

#NAIVE BAYES
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf1 = pickle.load(open('vectorizer.pkl','rb'))
model1 = pickle.load(open('model.pkl','rb'))


#LOGISTIC REGRESSION
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

tfidf2 = pickle.load(open('vectorizer_1.pkl','rb'))
model2 = pickle.load(open('model_1.pkl','rb'))

#SUPPORT VECTOR CLASSIFIER
tfidf3 = pickle.load(open('vectorizer_2.pkl','rb'))
model3 = pickle.load(open('model_2.pkl','rb'))



st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
option = st.selectbox('Select an option', ['NAIVE BAYES', 'LOGISTIC REGRESSION', 'SUPPORT VECTOR CLASSIFIER'])


if (option=='NAIVE BAYES'):
    if st.button('Predict'):
        #NAIVE BAYES
        # 1. preprocess
        transformed_sms1 = transform_text(input_sms)
        # 2. vectorize
        vector_input1 = tfidf1.transform([transformed_sms1])
        # 3. predict
        result1 = model1.predict(vector_input1)[0]
        NB = Image.open('img/NB.png')
        acc1=Image.open('img/NB_ACCURACY.png')
        # 4. Display
        if result1 == 1:
            st.header("Spam")
            spam_data(input_sms)
        else:
            st.header("Not Spam")
            ham_data(input_sms)
        st.image(acc1,width=550)
        st.image(NB, caption='MATRIX FOR NAIVE BAYES', width=550)

#LOGISTIC REGRESSION
if (option == 'LOGISTIC REGRESSION'):
    if st.button('Predict'):
        # 1. preprocess
        transformed_sms2 = text_preprocess(input_sms)
        transform_sms = stemmer(transformed_sms2)
        # 2. vectorize
        vector_input2 = tfidf2.transform([transform_sms])
        # 3. predict
        result2 = model2.predict(vector_input2)[0]
        LB = Image.open('img/LB.png')
        acc2=Image.open('img/LB_ACCURACY.png')
        # 4. Display
        if result2 == 1:
            st.header("Spam")
            spam_data(input_sms)
        else:
            st.header("Not Spam")
            ham_data(input_sms)
        st.image(acc2,width=550)
        st.image(LB, caption='MATRIX FOR LOGISTIC REGRESSION',width=550)

    #SUPPORT VECTOR CLASSIFIER
if (option == 'SUPPORT VECTOR CLASSIFIER'):
    if st.button('Predict'):
        # 1. preprocess
        transformed_sms3 = transform_text(input_sms)
        # 2. vectorize
        vector_input3 = tfidf3.transform([transformed_sms3])
        # 3. predict
        result3 = model3.predict(vector_input3)[0]
        SVC = Image.open('img/SVC.png')
        acc3=Image.open('img/SVC ACCURACY.png')
        # 4. Display
        if result3 == 1 :
            st.header("Spam")
            spam_data(input_sms)
        else:
            st.header("Not Spam")
            ham_data(input_sms)
        st.image(acc3,width=550)
        st.image(SVC, caption='MATRIX FOR SUPPORT VECTOR CLASSIFIER',width=550)

        #SPAM- XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> hXXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>>