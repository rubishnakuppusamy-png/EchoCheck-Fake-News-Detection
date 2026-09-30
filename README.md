# EchoCheck – AI-Based Fake News Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-EchoCheck%20Website-9333ea?style=for-the-badge&logo=googlechrome&logoColor=white)](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/)
[![GitHub Pages](https://img.shields.io/badge/Hosted%20on-GitHub%20Pages-22c55e?style=for-the-badge&logo=github&logoColor=white)](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/)
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.21%25-3b82f6?style=for-the-badge)](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/how-it-works.html)

---

### 🌐 Live Public Website
> **Click here to test the live application:**  
> 👉 **[https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/)**
> 
> *Runs 100% in your browser using client-side TF-IDF and calibrated SVM weights!*

---

## 1. Project Overview

**EchoCheck** is an academic machine learning web platform engineered to detect and analyze fabricated digital news content. It utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** feature engineering alongside **Calibrated Support Vector Machines (LinearSVC)** and **Logistic Regression** trained on the benchmark Kaggle Fake and Real News dataset.

- **Academic Domain:** Artificial Intelligence, Natural Language Processing, Machine Learning
- **Problem Statement:** Mitigating the spread of digital disinformation and sensationalist media by providing real-time veracity scores with calibrated confidence percentages.
- **Model Evaluation:**
  - **Calibrated Support Vector Machine:** 99.21% Accuracy | 0.9921 F1-Score
  - **Logistic Regression:** 98.87% Accuracy | 0.9888 F1-Score
- **Inference Speed:** < 5 ms in browser, < 30 ms on Python Flask
- **Persistence:** SQLite audit logging for server, `localStorage` for GitHub Pages

---

## 2. Technology Stack

- **Frontend:** HTML5, Modern Vanilla CSS3 (Soft Lavender, White, Light Pink aesthetic with Glassmorphism and Micro-Animations), Reactive Vanilla JavaScript, Chart.js
- **Backend:** Python 3.13, Flask 3.1 Framework
- **Machine Learning & NLP:** Scikit-Learn, TF-IDF Vectorizer (15,000 N-Gram features), LinearSVC with Platt Calibration (`CalibratedClassifierCV`), Logistic Regression
- **Browser-Based ML Engine:** Pure JavaScript vectorized dot product with Platt sigmoid calibration (`static/model/model_weights.json`)
- **Model Serialization:** Joblib
- **Database:** SQLite3 (`database/predictions.db`) / `localStorage`

---

## 3. Project Structure

```text
EchoCheck/
│
├── index.html                 # Main landing page for GitHub Pages
├── detect.html                # Detection workspace with animated circular score ring
├── how-it-works.html          # 5-Stage visual pipeline diagram & benchmark tables
├── about.html                 # Academic project context, dataset & tech stack
├── history.html               # Searchable audit log & Chart.js distribution
│
├── app.py                     # Flask application server and REST API endpoints
├── train_model.py             # Academic ML training, evaluation, and serialization script
├── text_preprocessing.py      # Shared text cleaning & linguistic feature diagnostics
├── db.py                      # SQLite database operations and audit logging
├── requirements.txt           # Python dependency specifications
├── Procfile                   # Cloud hosting configuration (Render / Railway)
├── README.md                  # Project documentation & live links
│
├── dataset/                   # Kaggle Fake and Real News CSVs (local only)
│   ├── Fake.csv               # 23,481 Disinformation articles
│   └── True.csv               # 21,417 Authenticated journalism articles
│
├── model/                     # Trained ML assets and metrics
│   ├── fake_news_model.pkl    # Serialized champion model (Calibrated LinearSVC)
│   ├── tfidf_vectorizer.pkl   # Serialized TF-IDF vectorizer (15,000 features)
│   └── model_metrics.json     # Empirical evaluation benchmarks & confusion matrix
│
├── database/                  # SQLite storage
│   └── predictions.db         # Persistent prediction audit records
│
├── templates/                 # Jinja2 HTML templates for Flask
│   ├── base.html
│   ├── index.html
│   ├── detect.html
│   ├── history.html
│   ├── how_it_works.html
│   └── about.html
│
└── static/                    # Frontend static assets
    ├── css/
    │   └── style.css          # Custom soft lavender & pink design system
    ├── js/
    │   └── main.js            # Client-side dynamic fetch, counters & hybrid ML engine
    ├── model/
    │   └── model_weights.json # Browser-ready compact ML weights & vocabulary
    └── images/
        └── hero-illustration.png # Visual graphic asset
```

---

## 4. How to Use the Application

### Option A: Open the Live Website (No Installation Required)
Simply navigate to:  
👉 **[https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/)**

### Option B: Run Locally with Python Flask
```powershell
# 1. Clone or navigate to the directory
cd "C:\Users\RUBISHNA .K\.gemini\antigravity-ide\scratch\EchoCheck"

# 2. Install dependencies
py -m pip install -r requirements.txt

# 3. Start the Flask server
py app.py

# 4. Open in your browser:
# http://127.0.0.1:5000
```

---

## 5. Machine Learning Methodology

1. **Ingestion & Dateline Stripping:** News agency tags (e.g. `(Reuters) -`, `(AP) -`) are systematically stripped using regex to prevent the model from overfitting solely to publisher signatures.
2. **Feature Extraction:** Sublinear TF-IDF scaling extracts unigrams and bigrams across 15,000 dimensions (`ngram_range=(1, 2)`).
3. **Platt Probability Calibration:** We use `CalibratedClassifierCV(LinearSVC())` with 3-fold cross-validation to produce true, mathematically sound posterior probabilities instead of uncalibrated distance margins.
4. **Linguistic Diagnostic Engine:** Computes uppercase ratios, punctuation frequency, and word complexity to produce human-readable diagnostic explanations for every prediction.

---

## 6. Academic Mini Project Credits
- **Project Name:** EchoCheck – AI-Based Fake News Detection System
- **Academic Year:** 2025–2026
- **Author:** Rubishna K
- **Dataset:** Kaggle ISOT Fake and Real News Dataset
- **Repository:** [https://github.com/rubishnakuppusamy-png/EchoCheck-Fake-News-Detection](https://github.com/rubishnakuppusamy-png/EchoCheck-Fake-News-Detection)
- **Live Demo:** [https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/](https://rubishnakuppusamy-png.github.io/EchoCheck-Fake-News-Detection/)
