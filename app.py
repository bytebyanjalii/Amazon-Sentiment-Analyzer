import os
import pickle

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ------------------ Load Model & Vectorizer ------------------
import os
import urllib.request
import pickle

MODEL_URL = "https://github.com/bytebyanjalii/Amazon-Sentiment-Analyzer/releases/download/v1.0-model/sentiment_model.pkl"
VECTORIZER_URL = "https://github.com/bytebyanjalii/Amazon-Sentiment-Analyzer/releases/download/v1.0-model/tfidf_vectorizer.pkl"

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Download model if not present (for Streamlit Cloud)
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

if not os.path.exists(VECTORIZER_PATH):
    urllib.request.urlretrieve(VECTORIZER_URL, VECTORIZER_PATH)

model = pickle.load(open(MODEL_PATH, "rb"))
tfidf = pickle.load(open(VECTORIZER_PATH, "rb"))


stop_words = set(stopwords.words('english'))

# ------------------ Text Cleaning Function ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# ------------------ Prediction Function ------------------
def predict_sentiment(review):
    review_clean = clean_text(review)
    review_vec = tfidf.transform([review_clean])
    prediction = model.predict(review_vec)[0]
    return prediction

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Amazon Sentiment Analyzer", layout="centered")

st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Paste Amazon product reviews below **(one review per line)** and analyze overall product feedback.")

review_text = st.text_area(
    "✍️ Paste Amazon Reviews Here",
    height=260,
    placeholder="Write your Amazon reviews here..."
)

if st.button("Analyze Product Reviews"):
    if review_text.strip() == "":
        st.warning("Please paste at least one review.")
    else:
        reviews = review_text.split("\n")

        positive_count = 0
        negative_count = 0
        results = []

        for review in reviews:
            if review.strip() == "":
                continue

            sentiment = predict_sentiment(review)
            results.append((review, sentiment))

            if sentiment == "positive":
                positive_count += 1
            else:
                negative_count += 1

        total_reviews = positive_count + negative_count

        # -------- Summary --------
        st.subheader("📊 Analysis Summary")
        st.write(f"**Total Reviews:** {total_reviews}")
        st.write(f"✅ **Positive Reviews:** {positive_count}")
        st.write(f"❌ **Negative Reviews:** {negative_count}")

        if positive_count > negative_count:
            st.success("🟢 Overall Product Feedback: **POSITIVE**")
        else:
            st.error("🔴 Overall Product Feedback: **NEGATIVE**")

        # -------- Individual Review Results --------
        st.subheader("📝 Individual Review Analysis")
        for review, sentiment in results:
            if sentiment == "positive":
                st.markdown(f"✅ **Positive** — {review}")
            else:
                st.markdown(f"❌ **Negative** — {review}")

