import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Download NLTK Data ---------------- #

@st.cache_resource
def download_nltk():
    nltk.download("punkt")
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
    message = nltk.word_tokenize(message)

    y = [ps.stem(word) for word in message if word.isalnum() and word not in stop_words]

    return " ".join(y)

# ---------------- Streamlit UI ---------------- #

st.title("📧 Spam Detection App")

msg = st.text_input("Enter your message")

if msg:

    transformed_msg = transform_message(msg)

    vector_input = tfidf.transform([transformed_msg]).toarray()

    prediction = model.predict(vector_input)

    st.write("Prediction:", prediction[0])

    if prediction[0] == 1:
        st.error("⚠️ Spam Message")
    else:
        st.success("✅ Not Spam")

# # ---------------- Sidebar ---------------- #

# st.sidebar.title("📊 Sidebar")

# sidebar_option = st.sidebar.selectbox("Choose option", ["Prediction", "Charts"])

# if sidebar_option == "Charts":

#     st.subheader("Dataset Correlation Heatmap")

#     fig, ax = plt.subplots()

#     sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)

#     st.pyplot(fig)
