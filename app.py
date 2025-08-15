import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import time
# Unduh stopwords jika belum
nltk.download('stopwords')

# === Load Model dan Vectorizer ===
with open('model/naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# === Fungsi Preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words if not w in stop_words]
    return ' '.join(words)

# === Tampilan GUI Streamlit ===
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Email Spam Detector")
st.subheader("Masukkan teks email di bawah ini:")

email_input = st.text_area("Teks Email", height=200)

if st.button("Deteksi"):
    if email_input.strip() == "":
        st.warning("Silakan masukkan teks email terlebih dahulu.")
    else:
        start_time = time.time()  # Mulai hitung waktu

        cleaned = clean_text(email_input)
        vectorized = vectorizer.transform([cleaned])
        prediction_proba = model.predict_proba(vectorized)[0]
        prediction = model.predict(vectorized)[0]

        end_time = time.time()  # Waktu selesai
        elapsed_time = round(end_time - start_time, 4)  # Waktu respon dalam detik

        spam_percent = round(prediction_proba[1] * 100, 2)
        safe_percent = round(prediction_proba[0] * 100, 2)

        if prediction == 1:
            st.error(f"🚫 Email ini terdeteksi sebagai SPAM ({spam_percent}%)")
        else:
            st.success(f"✅ Email ini terdeteksi sebagai Aman ({safe_percent}%)")

        st.markdown("---")
        st.write("**Probabilitas Klasifikasi:**")
        st.write(f"- Spam: {spam_percent}%")
        st.write(f"- Aman: {safe_percent}%")

        st.markdown("---")
        st.info(f"⏱️ Waktu proses klasifikasi: **{elapsed_time} detik**")