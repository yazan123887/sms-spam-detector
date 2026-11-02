import urllib.request
import zipfile
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import streamlit as st

# Dataset download URL
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Download dataset
if not os.path.exists("spam.zip"):
    urllib.request.urlretrieve(url, "spam.zip")

# Extract dataset
with zipfile.ZipFile("spam.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

# Load dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))


print(df.head())
print("\nDataset size:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# Streamlit interface
st.title("SMS Spam Detector")
st.write("Enter a message and let the ML model classify it.")

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

user_message = st.text_area(
    "Enter your message:",
    placeholder="Congratulations! You won a free iPhone!"
)

if st.button("Analyze Message"):

    if user_message.strip():

        # Predict message category
        prediction = model.predict([user_message])[0]

        # Get prediction probability
        probabilities = model.predict_proba([user_message])[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.error("SPAM DETECTED!")
        else:
            st.success("LEGITIMATE MESSAGE")

        st.write(f"Prediction: {prediction.upper()}")
        st.write(f"Model confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter a message first.")