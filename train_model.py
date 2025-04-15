# train_model.py
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils import is_gibberish

import nltk
nltk.download('stopwords')

# Load data
df = pd.read_csv("email_classification.csv")

# Rename column if needed
if 'email' in df.columns:
    df.rename(columns={'email': 'message'}, inplace=True)

print("Total messages before filtering:", len(df))

# Remove gibberish
df["is_gibberish"] = df["message"].apply(is_gibberish)
print("Gibberish samples:\n", df[df["is_gibberish"]]["message"].head(10))
df = df[~df["is_gibberish"]].reset_index(drop=True)

print("Remaining after gibberish filtering:", len(df))

# Preprocess
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(text):
    text = re.sub(r"\W", " ", text)
    text = text.lower()
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

df["message_clean"] = df["message"].apply(preprocess)

# Feature extraction
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df["message_clean"]).toarray()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_bow)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

y = df["label"]

# Train model
model = ExtraTreesClassifier()
model.fit(X_pca, y)

# Save models
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete.")
print("✅ Models saved successfully.")
print("Total messages after preprocessing:", len(df))