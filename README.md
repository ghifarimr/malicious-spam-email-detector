# 📧 Malicious Email Spam Detector

> A web-based Machine Learning application for detecting malicious spam emails using Natural Language Processing (NLP) and the Naive Bayes algorithm.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)

---

## 📖 Overview

Email remains one of the most widely used communication channels, but it is also one of the primary targets for cyber threats such as phishing, fraud, malware distribution, and other malicious spam campaigns.

This project implements a **Machine Learning-based email spam detection system** capable of classifying email text into **Spam** or **Non-Spam** categories.

The application utilizes:

- Natural Language Processing (NLP)
- TF-IDF Feature Extraction
- Multinomial Naive Bayes Classification
- Streamlit Web Application

The trained model can classify new email text in real-time through a simple web interface.

---

## 🎯 Project Objectives

- Build an email spam detection model using Machine Learning.
- Apply Natural Language Processing techniques for text preprocessing.
- Evaluate multiple train-test split scenarios.
- Deploy the trained model into an interactive web application.
- Demonstrate real-time email spam classification.

---

## 🚀 Features

- Email text classification
- Real-time prediction
- Spam probability score
- Safe email probability score
- Fast prediction (~0.005 seconds)
- Simple and responsive Streamlit interface

---

## 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Machine Learning | Scikit-learn |
| Algorithm | Multinomial Naive Bayes |
| NLP | NLTK |
| Feature Extraction | TF-IDF Vectorizer |
| Web Framework | Streamlit |
| Serialization | Pickle |
| Dataset | Kaggle Email Spam Dataset + HuggingFace Validation Dataset |

---

## 🧠 Machine Learning Workflow

```
Dataset
      │
      ▼
Text Preprocessing
(Cleaning
Tokenization
Stopword Removal
Stemming)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train/Test Split
(80:20)
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Model Evaluation
      │
      ▼
Model Deployment
(Streamlit)
```

---

## 📊 Model Performance

The model was evaluated using three train-test split scenarios.

| Train-Test Split | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| 80 : 20 | **92.01%** | **96.99%** | **82.09%** | **88.92%** |
| 70 : 30 | 91.66% | 96.66% | 81.30% | 88.32% |
| 60 : 40 | 91.42% | 96.31% | 81.00% | 87.99% |

The **80:20 split** achieved the best overall performance and was selected as the final model.

---

## ✅ Validation

### Internal Validation

- Accuracy: 100% (sample testing)
- Average Prediction Time: **0.00538 seconds**

### External Validation

The deployed application was further tested using an external dataset that had never been used during model training.

Results demonstrated that the model maintained consistent predictions with good generalization capability.

---

## 🌐 Live Demo

**Streamlit App**

https://malicious-spam-email-detector.streamlit.app/

---

## 💻 Installation

Clone this repository

```bash
git clone https://github.com/yourusername/malicious-email-spam-detector.git
```

Go into project folder

```bash
cd malicious-email-spam-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📚 Research Highlights

This project was developed as an undergraduate thesis:

**Implementation of the Naive Bayes Algorithm for Web-Based Malicious Email Spam Detection**

Key contributions include:

- End-to-end NLP pipeline
- Feature engineering using TF-IDF
- Comparative evaluation of train-test split ratios
- Streamlit deployment
- Internal & external validation

---

## 🔮 Future Improvements

- Gmail API integration
- URL detection
- Attachment analysis
- Metadata-based detection
- Deep Learning implementation
- Transformer-based models (BERT)
