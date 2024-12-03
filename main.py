import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense


# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

# load the pre-trained n=model with relu activbation
model = load_model("simple_rnn_imdb.h5")

# helper function 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])


# function to preprocess user 
def preprocess_text(text):
    words =text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review  =sequence.pad_sequences([encoded_review],maxlen= 500)
    return padded_review

# prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction  = model.predict(preprocessed_input)
    sentiment  = 'postive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


# stremit app
import streamlit as st

st.title('IMDB movie review sentiment analysis')
st.write('enter a movie review toi classify it as postive or negative. ')

# user input 
user_input =st.text_area('Movie Review')

# user nput
if st.button('Classify'):
    preprocess_input =preprocess_text(user_input)

    # make prediction 
    prediction  = model.predict(preprocess_input)
    sentiment  = 'Postive' if prediction[0][0]>0.5 else 'Negative'

    # desplay the results
    st.write(f'sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
