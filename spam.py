import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import pandas as pd

@st.cache_resource
def download_nltk():
    nltk.download("stopwords")

download_nltk()

# ---------------- Load Dataset ---------------- #

df = pd.read_csv("test.csv", encoding="iso-8859-1")

# ---------------- Initialize Stemmer ---------------- #

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ---------------- Load Model ---------------- #

model = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")

# ---------------- Text Preprocessing ---------------- #

def transform_message(message):
    message = message.lower()

    message = nltk.tokenize.wordpunct_tokenize(message)

    words = [
        ps.stem(word)
        for word in message
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(words)

# ---------------- Streamlit code ---------------- #

st.title("📧 Spam Email Detection App")

st.write("Enter a message below to check if it is **Spam or Not Spam**.")

msg = st.text_input("Enter your message")

if msg:

    transformed_msg = transform_message(msg)

    vector_input = tfidf.transform([transformed_msg]).toarray()

    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.error("⚠️ Spam Message")
    else:
        st.success("✅ Not Spam")

    st.write("Prediction Value:", prediction[0])
