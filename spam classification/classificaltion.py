import streamlit as st
import pickle as pkl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import string
import spacy

nlp = spacy.load("en_core_web_sm")


sw = stopwords.words('english') # get all the stop words
punc = string.punctuation

# stemmer
ps = PorterStemmer()
sbs = SnowballStemmer(language='english')

import os

vectorizer_path = os.path.abspath("vectorizer.pkl")
model_path = os.path.abspath("model.pkl")

print(vectorizer_path)
print(model_path)

tfidf = pkl.load(open(vectorizer_path,'rb'))
model = pkl.load(open(model_path,'rb'))

# 1. preprocess
# 2. vectorize
# 3. predict
# 4. Display

# implement lemetizing . 
def transorm_text_3(text):

    # convert into tokens 
    text = text.lower()
    text = word_tokenize(text)
    
    # remove special characters  
    y = []
    for i in text :
        if str(i).isalnum():
            y.append(i)
    
    # remove stropwords and punctuations 
    text = y[:]
    y.clear()

    # removing the stop words and punctions
    for i in text:
        if i not in sw and i not in punc :
            y.append(i)

    # use stemming
    text = y[:]
    y.clear()

    for i in text :
        y.append(sbs.stem(i))

    text = " ".join(y)

    # Process the text
    doc = nlp(text)

    lemmas = [token.lemma_ for token in doc]

    return " ".join(lemmas)


st.title('Email/Sms Spam classifier')
input_sms = st.text_input('Enter the message')
if st.button('Predict'):
    transormed_sms = transorm_text_3(input_sms)
    vector_input = tfidf.transform([transormed_sms])
    result = model.predict(vector_input)[0]
    if result == 1 :
        st.header('spam')
    else:
        st.header('not spam')
