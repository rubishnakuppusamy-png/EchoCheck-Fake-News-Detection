# EchoCheck – AI-Based Fake News Detection System

EchoCheck is a full-stack academic machine learning web platform engineered to detect and analyze fabricated digital news content. It utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** feature engineering alongside **Calibrated Support Vector Machines (LinearSVC)** and **Logistic Regression** trained on the benchmark Kaggle Fake and Real News dataset.

---

## 1. Project Overview

- **Academic Domain:** Artificial Intelligence, Natural Language Processing, Machine Learning
- **Problem Statement:** Mitigating the spread of digital disinformation and sensationalist media by providing real-time veracity scores with calibrated confidence percentages.
- **Model Evaluation:**
  - **Calibrated Support Vector Machine:** 99.21% Accuracy | 0.9921 F1-Score
  - **Logistic Regression:** 98.87% Accuracy | 0.9888 F1-Score
- **Inference Speed:** < 30 ms per article
- **Persistence:** SQLite audit logging for analysis history

---

## 2. Technology Stack

- **Frontend:** HTML5, Modern Vanilla CSS3 (Soft Lavender, White, Light Pink aesthetic with Glassmorphism and Micro-Animations), Reactive Vanilla JavaScript, Chart.js
- **Backend:** Python 3.13, Flask 3.1 Framework
- **Machine Learning & NLP:** Scikit-Learn, TF-IDF Vectorizer (15,000 N-Gram features), LinearSVC with Platt Calibration (`CalibratedClassifierCV`), Logistic Regression
- **Model Serialization:** Joblib
- **Database:** SQLite3 (`database/predictions.db`)

---

## 3. Project Structure

```text
EchoCheck/
│
├── app.py                     # Flask application server and REST API endpoints
├── train_model.py             # Academic ML training, evaluation, and serialization script
├── text_preprocessing.py      # Shared text cleaning & linguistic feature extraction
├── db.py                      # SQLite database operations and audit logging
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation
│
├── dataset/                   # Kaggle Fake and Real News CSVs
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
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Shared base layout with navbar and footer
│   ├── index.html             # Landing page with hero, features & disclaimer
│   ├── detect.html            # News detection workspace with animated results card
│   ├── history.html           # Prediction history table with search & filters
│   ├── how_it_works.html      # 5-Stage NLP pipeline explanation & benchmarks
│   └── about.html             # Academic project context, dataset & tech stack
│
└── static/                    # Frontend static assets
    ├── css/
    │   └── style.css          # Custom soft lavender & pink design system
    ├── js/
    │   └── main.js            # Dynamic frontend interactions & API requests
    └── images/
        └── hero-illustration.png # Visual graphic asset
```

---

## 4. Installation & Setup Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PowerShell (Windows) or Terminal (macOS / Linux)

### Step 1: Open PowerShell in the Project Directory
```powershell
cd "C:\Users\RUBISHNA .K\.gemini\antigravity-ide\scratch\EchoCheck"
```

### Step 2: Install Python Dependencies
```powershell
py -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model (Optional if pre-trained)
The project comes with a pre-trained model and TF-IDF vectorizer in `model/`. To retrain or evaluate again on the full Kaggle dataset:
```powershell
py train_model.py
```

### Step 4: Launch the Web Application
```powershell
py app.py
```

### Step 5: Open in Your Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## 5. Web Application Pages & Endpoints

### User Interface Pages:
- `GET /` — **Home Landing Page:** Hero banner, key features, architecture overview, and academic disclaimer.
- `GET /detect` — **Detection Workspace:** Article textarea with live character counter, 1-click curated samples, animated loading sequence, circular progress confidence ring, and linguistic metrics.
- `GET /history` — **Prediction History:** SQLite audit log with KPI cards, real/fake search filtering, view full text modal, and clear history option.
- `GET /how-it-works` — **Methodology & Pipeline:** Detailed 5-stage NLP breakdown and real confusion matrix benchmarks.
- `GET /about` — **Project Overview:** Academic scope, dataset details, tech stack, and ethical limitations.

### REST API Endpoints:
- `POST /predict` — Evaluates submitted news article text.
  - **Request Body (JSON):** `{"text": "<article text>"}`
  - **Response (JSON):** `{"success": true, "label": "REAL"|"FAKE", "confidence": 99.2, "real_prob": 99.2, "fake_prob": 0.8, "signals": {...}, "explanation": "..."}`
- `GET /api/history` — Retrieves JSON history records with optional search parameter `?q=`.
- `DELETE /api/history` — Clears prediction history.
- `GET /api/samples` — Returns pre-curated test articles.
- `GET /api/metrics` — Returns model benchmark scores from `model_metrics.json`.

---

## 6. Machine Learning Methodology

1. **Ingestion & Dateline Stripping:** News agency tags (e.g. `(Reuters) -`, `(AP) -`) are systematically stripped using regex to prevent the model from overfitting solely to publisher signatures.
2. **Feature Extraction:** Sublinear TF-IDF scaling extracts unigrams and bigrams across 15,000 dimensions (`ngram_range=(1, 2)`).
3. **Platt Probability Calibration:** We use `CalibratedClassifierCV(LinearSVC())` with 3-fold cross-validation to produce true, mathematically sound posterior probabilities instead of uncalibrated distance margins.
4. **Linguistic Diagnostic Engine:** Computes uppercase ratios, punctuation frequency, and word complexity to produce human-readable diagnostic explanations for every prediction.

---

## 7. Academic Mini Project Credits
- **Project Name:** EchoCheck – AI-Based Fake News Detection System
- **Academic Year:** 2025–2026
- **Dataset:** Kaggle ISOT Fake and Real News Dataset
