# app.py
import streamlit as st
import joblib
import pandas as pd
from utils import is_gibberish, detect_alert
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(text):
    text = re.sub(r"\W", " ", text)
    text = text.lower()
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

def classify_message(message):
    if is_gibberish(message):
        return "🔵 Gibberish"
    if detect_alert(message):
        return "🟡 Alert"
    clean = preprocess(message)
    X = vectorizer.transform([clean]).toarray()
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    pred = model.predict(X_pca)[0]
    if pred == "spam":
        return "🔴 Spam"
    return "🟢 Not Spam"

# Streamlit UI
st.set_page_config(page_title="SMS Classifier", layout="centered")
st.title("📱 SMS & Email Classifier")

st.write("Classifies messages as: **Not Spam**, **Spam**, **Alert**, or **Gibberish**")

user_input = st.text_area("Enter one or more messages (one per line):", height=200)

if st.button("Classify"):
    messages = user_input.strip().split("\n")
    results = []
    for msg in messages:
        if msg.strip():
            category = classify_message(msg)
            results.append((msg, category))

    st.subheader("📊 Results")
    for msg, cat in results:
        st.markdown(f"**{cat}** — `{msg}`")
    st.success("Classification complete!")
else:   
    st.info("Click the button to classify the messages.")   