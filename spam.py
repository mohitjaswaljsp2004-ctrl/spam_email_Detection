import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
df= pd.read_csv('test.csv',  encoding='iso-8859-1')

ps = PorterStemmer()

# Load model and vectorizer

model = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")

def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    
    y = [ps.stem(word) for word in message if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(y)

st.title("Text Classification App")
msg = st.text_input("Enter your message")
tfidf = joblib.load("vectorizer.pkl")

if msg:
    transformed_msg = transform_message(msg)
    vector_input = tfidf.transform([transformed_msg]).toarray()
    prediction = model.predict(vector_input)
   
    st.write("Prediction:", prediction[0])

    if prediction==1:
        st.write("⚠️Spam")
    else:
        st.write("Not Spam")


# sidebar :-

# st.sidebar.title("Sidebar")
# st.sidebar.write("This is a Sidebar")

# # sidebar inputs
# sidebar_option=st.sidebar.selectbox("choose option",["Prediction","Charts"])

# fig, ax = plt.subplots()
# sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
# st.pyplot(fig)

# sns.show()


# sidebar_slider=st.sidebar.slider("Sidebar slider",0,100,50)


## Sidebar with form

# with st.sidebar:
#     st.header("Settings")
#     filter_option=st.selectbox("Filter by",["All","CategoryA","CategoryB"])
#     data_range=st.date_input("Date range",[])