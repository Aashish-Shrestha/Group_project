import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
@st.cache_data
def load_data():
    spam_df = pd.read_csv("Spams.csv")
    spam_df['spam'] = spam_df['v1'].apply(lambda x: 1 if x == 'spam' else 0)
    return spam_df

@st.cache_data
def preprocess_data(spam_df, max_words=10000, max_length=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(spam_df['v2'])
    sequences = tokenizer.texts_to_sequences(spam_df['v2'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return tokenizer, padded_sequences, spam_df['spam']

# Build the Keras model
def build_model(max_words=10000, max_length=100):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=32, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load data and preprocess
spam_df = load_data()
max_words = 10000
max_length = 100
tokenizer, padded_sequences, labels = preprocess_data(spam_df, max_words, max_length)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build and train the model
model = build_model(max_words, max_length)
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32)  # Reduced epochs to 5

# Streamlit app interface
st.title("Spam Detection")


user_input = st.text_area("Enter your message:")
if st.button("Classify"):
    if user_input.strip():
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')
        prediction = model.predict(input_padded)[0][0]
        result = "Spam" if prediction > 0.5 else "Ham"
        st.write(f"### The message is classified as: {result}")
        st.write(f"Spam probability: {prediction:.2f}")
    else:
        st.write("Please enter a valid message.")

# Model evaluation
if st.checkbox("Show Model Performance"):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"Test Loss: {loss:.2f}")
    st.write(f"Test Accuracy: {accuracy:.2f}")
