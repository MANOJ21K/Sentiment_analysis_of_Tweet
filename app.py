import streamlit as st
import pickle
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the Word2Vec model as a KeyedVectors object
word2vec_model = KeyedVectors.load('Word2Vec-twitter-100')

# Load the tokenizer
with open('Tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the TF-Model
training_model = load_model('Sentiment-RNN')

# Define the Streamlit app
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet, and we'll predict its sentiment as either Positive or Negative.")

# User input for a new tweet
user_input = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if not user_input:
        st.warning("Please enter a tweet before analyzing sentiment.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize and pad the user's input tweet
            # input_length = st.slider("Select Input Length", min_value=10, max_value=100, value=60)
            user_input_tokens = tokenizer.texts_to_sequences([user_input])
            user_input_tokens_padded = pad_sequences(user_input_tokens, maxlen=60)

            # Predict sentiment for the user's input tweet
            sentiment_probabilities = training_model.predict(user_input_tokens_padded)

            # Determine the predicted sentiment label
            predicted_sentiment = "Positive" if sentiment_probabilities[0][0] > 0.5 else "Negative"

            # Display the predicted sentiment and probability
            st.write(f"Predicted Sentiment: {predicted_sentiment} ({sentiment_probabilities[0][0]:.2f})")

        # Optionally, display the original tweet
        #st.subheader("Original Tweet:")
        #st.write(user_input)
