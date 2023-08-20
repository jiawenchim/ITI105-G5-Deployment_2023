from textblob import TextBlob
import pandas as pd
import streamlit as st
!pip install cleantext
import cleantext
from sklearn.ensemble import RandomForestClassifier
import pickle

filename = "cyberbullymodel_min_df35.sav"
vect_file = "vectorizer.sav"
model = pickle.load(open(filename, "rb"))
vectorizer = pickle.load(open(vect_file, "rb"))

def predict(x): 
    x_str = str(x)
    text_to_vect = vectorizer.transform([x_str])
    result = model.predict(text_to_vect)
    return result

st.header("Cyberbully Detection")
with st.expander("Analyze Text"): 
    text = st.text_input("Text here: ")
    if text: 
        prediction = predict(text)
        if prediction[0] == 1: 
            st.write("This is a cyberbully message.")
        elif prediction[0] == 0:
            st.write("This is not a cyberbully message." 
        
