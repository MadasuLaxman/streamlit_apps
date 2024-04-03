import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import re
import emoji
import pandas as pd

def basic_pp(x,emoj="T"):
    x = x.lower() #converting into lower case
    x = re.sub("<.*?>"," ", x ) #removing html tags
    x = re.sub("http[s]?://.+?\S+"," ",x) #removing urls
    x = re.sub("#\S+"," ",x) #removing hashtags
    x = re.sub("@\S+"," ",x) #removing mentions
    if emoj=="T":
        x =emoji.demojize(x) #converting emoji to text
    x = re.sub("[]\:.\*'\-#$%^&)(0-9]"," ",x) #removing unwanted charecters
    x = re.sub("[^a-zA-Z.]", " ", x)  # removing non-alphabetic characters
    return x
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    tokens = [token.lower() for token in tokens if token not in string.punctuation and token not in stop_words]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

df = pd.read_csv(r"C:\Users\madas\Downloads\fakenews.csv")
df["text"] = df["text"].apply(basic_pp,args=("T"))
X = df['text'].apply(preprocess_text)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Load your models and TF-IDF vectorizer
knn_model = pickle.load(open(r"C:\Users\madas\Downloads\knn.pkl","rb"))
bernoulli_nb_model = pickle.load(open(r"C:\Users\madas\Downloads\Bnb.pkl","rb"))
multinomial_nb_model = pickle.load(open(r"C:\Users\madas\Downloads\Mnb.pkl","rb"))

# Streamlit app
st.title('Fake News Prediction App')

# Input text
input_text = st.text_input('Enter the news text:')

if st.button('Predict'):
    # Preprocess input text
    processed_text = preprocess_text(input_text)
    
    # Vectorize processed text
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    
    # Make predictions using the trained models
    knn_pred = knn_model.predict(text_vectorized)
    bernoulli_nb_pred = bernoulli_nb_model.predict(text_vectorized)
    multinomial_nb_pred = multinomial_nb_model.predict(text_vectorized)
    
    # Display predictions
    st.write('KNN Prediction:', 'Real News' if knn_pred[0] == 0 else 'Fake News')
    st.write('Bernoulli NB Prediction:', 'Real News' if bernoulli_nb_pred[0] == 0 else 'Fake News')
    st.write('Multinomial NB Prediction:', 'Real News' if multinomial_nb_pred[0] == 0 else 'Fake News')
