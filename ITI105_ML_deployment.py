{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "edde5c2e-5d8b-4b46-bc41-91a8ae94eee7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: cleantext in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (1.1.4)\n",
      "Requirement already satisfied: nltk in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from cleantext) (3.7)\n",
      "Requirement already satisfied: tqdm in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from nltk->cleantext) (4.64.1)\n",
      "Requirement already satisfied: joblib in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from nltk->cleantext) (1.1.1)\n",
      "Requirement already satisfied: regex>=2021.8.3 in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from nltk->cleantext) (2022.7.9)\n",
      "Requirement already satisfied: click in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from nltk->cleantext) (8.0.4)\n",
      "Requirement already satisfied: colorama in c:\\users\\jwchi\\anaconda3\\lib\\site-packages (from click->nltk->cleantext) (0.4.6)\n"
     ]
    }
   ],
   "source": [
    "from textblob import TextBlob\n",
    "import pandas as pd\n",
    "import streamlit as st\n",
    "!pip install cleantext\n",
    "import cleantext\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0a2ad771-ab30-4918-824d-6431f6e2937f",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = \"cyberbullymodel_min_df35.sav\"\n",
    "vect_file = \"vectorizer.sav\"\n",
    "model = pickle.load(open(filename, \"rb\"))\n",
    "vectorizer = pickle.load(open(vect_file, \"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "41406290-c6ba-4c07-8e89-1d09bb66c687",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(x): \n",
    "    x_str = str(x)\n",
    "    text_to_vect = vectorizer.transform([x_str])\n",
    "    result = model.predict(text_to_vect)\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57a11a8c-9e57-419d-a6c8-3b26d47eeb6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.header(\"Cyberbully Detection\")\n",
    "with st.expander(\"Analyze Text\"): \n",
    "    text = st.text_input(\"Text here: \")\n",
    "    if text: \n",
    "        prediction = predict(text)\n",
    "        if prediction[0] == 1: \n",
    "            st.write(\"This is a cyberbully message.\")\n",
    "        elif prediction[0] == 0:\n",
    "            st.write(\"This is not a cyberbully message.\" \n",
    "        "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
