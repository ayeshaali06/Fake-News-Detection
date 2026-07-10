
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

# Load model and tokenizer
model = load_model("fake_news_model.keras")

import traceback

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    st.success("Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Tokenizer Error: {e}")
    st.code(traceback.format_exc())
    st.stop()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detection System")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    news = clean_text(news)
    seq = tokenizer.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen=300)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        st.success(f"✅ Real News ({prediction*100:.2f}%)")
    else:
        st.error(f"❌ Fake News ({(1-prediction)*100:.2f}%)")
