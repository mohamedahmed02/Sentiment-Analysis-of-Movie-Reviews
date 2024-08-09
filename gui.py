import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import streamlit as st
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from spellchecker import SpellChecker
import random
import joblib
import string
from nltk.corpus import stopwords
import os
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, learning_curve,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from io import StringIO
import base64

# Function to perform sentiment analysis

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]

    # Remove digits and single-character words
    tokens = [word for word in tokens if not (word.isdigit() or len(word) == 1)]

    # Spell correction
    # spell = SpellChecker()
    # tokens = [spell.correction(word) for word in tokens]

    # Lemmatize words
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) if word is not None else '' for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens
# Function to generate word cloud
# Function to generate word cloud
# Function to generate word cloud
def load_model_and_vectorizer():
    classifier = joblib.load('svm_model.pkl')  # Update with your actual model filename
    # tfidf_vect = joblib.load('tfidf_vectorizer.pkl')
    tfidf_vect = joblib.load('tfidf_vectorizer.pkl').set_params(max_features=37300)
    # Update with your actual vectorizer filename
    return classifier, tfidf_vect
def get_class_label(prediction):
    # You can define your custom mapping here if needed
    class_names = {0: "Negative", 1: "Positive"}  # Example mapping
    return class_names.get(prediction, "Unknown")
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("wordcloud.png")
    plt.close()

# Main function
def main():
    # Center align titles
    st.markdown("<div style='text-align: center;'><h1>Sentiment Analysis of Movie Reviews</h1></div>",
                unsafe_allow_html=True)
    # Displaying the popcorn image
    image = Image.open("popCorn.png")
    st.image(image, caption='Popcorn', use_column_width=True)

    st.write("Upload the text file for sentiment analysis:")
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("File Uploaded Successfully!")
        Txt_P=preprocess_text(text)
        input_text_joined = ' '.join(Txt_P)
        classifier, tfidf_vect = load_model_and_vectorizer()

        # Transform the preprocessed input text
        input_tfidf = tfidf_vect.transform([input_text_joined])

        # Displaying prediction
        # prediction = predict_sentiment(input_tfidf)
        predictions = classifier.predict(input_tfidf)
        predicted_label = get_class_label(predictions[0])
        st.write("Prediction Label:", predicted_label)

        # Generating and displaying word cloud
        generate_wordcloud(text)
        st.image("wordcloud.png", use_column_width=True)

if __name__ == "__main__":
    main()