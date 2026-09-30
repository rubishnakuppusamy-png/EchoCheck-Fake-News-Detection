"""
EchoCheck - Model Training and Evaluation Script
Academic Machine Learning Pipeline for Fake vs. Real News Detection.

Datasets: Kaggle Fake and Real News Dataset (Fake.csv & True.csv)
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)
Classifiers Evaluated:
  1. Logistic Regression
  2. Support Vector Machine (LinearSVC with Calibrated Probabilities)

Outputs:
  - model/fake_news_model.pkl
  - model/tfidf_vectorizer.pkl
  - model/model_metrics.json
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import shared text cleaner
from text_preprocessing import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(max_samples_per_class: int = 12000):
    """
    Loads Fake.csv and True.csv from dataset directory.
    If full dataset is available, samples a balanced subset for fast, optimal convergence.
    """
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(
            f"Dataset files not found. Expected:\n  - {FAKE_CSV}\n  - {TRUE_CSV}\n"
            "Please ensure Fake.csv and True.csv are placed inside the 'dataset' directory."
        )

    print(f"Loading Fake.csv from {FAKE_CSV}...")
    df_fake = pd.read_csv(FAKE_CSV)
    print(f"Loading True.csv from {TRUE_CSV}...")
    df_true = pd.read_csv(TRUE_CSV)

    # Assign binary labels: 0 = FAKE, 1 = REAL
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine title and body text for richer contextual features
    df_fake["full_text"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["full_text"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    # Stratified balance sampling if requested
    if max_samples_per_class and len(df_fake) > max_samples_per_class:
        df_fake = df_fake.sample(n=max_samples_per_class, random_state=42)
    if max_samples_per_class and len(df_true) > max_samples_per_class:
        df_true = df_true.sample(n=max_samples_per_class, random_state=42)

    df_combined = pd.concat([df_fake[["full_text", "label"]], df_true[["full_text", "label"]]], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Total samples loaded: {len(df_combined)} ({len(df_fake)} Fake, {len(df_true)} Real)")
    return df_combined


def train_and_evaluate():
    start_total_time = time.time()
    print("=" * 65)
    print(" ECHOCHECK ML PIPELINE: TRAINING & EVALUATION")
    print("=" * 65)

    # 1. Load data
    df = load_dataset(max_samples_per_class=12000)

    # 2. Text preprocessing
    print("\n[Stage 1/5] Preprocessing text corpus using clean_text()...")
    clean_start = time.time()
    df["cleaned_text"] = df["full_text"].apply(clean_text)
    # Filter out empty or negligible strings
    df = df[df["cleaned_text"].str.split().str.len() > 10].reset_index(drop=True)
    print(f"Text cleaned in {time.time() - clean_start:.2f}s. Valid articles: {len(df)}")

    # 3. Train / Test Split
    print("\n[Stage 2/5] Splitting data into 80% train and 20% test sets...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )
    print(f"Training set: {len(X_train_raw)} | Test set: {len(X_test_raw)}")

    # 4. TF-IDF Feature Extraction
    print("\n[Stage 3/5] Extracting TF-IDF Features (unigram & bigram)...")
    tfidf_start = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.85,
        stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"TF-IDF Matrix: {X_train_tfidf.shape[0]} articles x {X_train_tfidf.shape[1]} features (built in {time.time() - tfidf_start:.2f}s)")

    # 5. Train & Evaluate Models
    print("\n[Stage 4/5] Training Classifiers...")

    # Classifier 1: Logistic Regression
    print(" -> Training Logistic Regression (C=2.0, max_iter=1000)...")
    lr_start = time.time()
    lr_model = LogisticRegression(C=2.0, max_iter=1000, random_state=42, solver="lbfgs")
    lr_model.fit(X_train_tfidf, y_train)
    lr_train_time = time.time() - lr_start

    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()

    print(f"    Logistic Regression -> Accuracy: {lr_acc * 100:.2f}%, F1: {lr_f1:.4f} (trained in {lr_train_time:.2f}s)")

    # Classifier 2: Support Vector Machine (LinearSVC with CalibratedClassifierCV)
    print(" -> Training Support Vector Machine (Calibrated LinearSVC)...")
    svm_start = time.time()
    base_svm = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm_model.fit(X_train_tfidf, y_train)
    svm_train_time = time.time() - svm_start

    svm_preds = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds)
    svm_prec = precision_score(y_test, svm_preds)
    svm_rec = recall_score(y_test, svm_preds)
    svm_cm = confusion_matrix(y_test, svm_preds).tolist()

    print(f"    SVM (Calibrated)    -> Accuracy: {svm_acc * 100:.2f}%, F1: {svm_f1:.4f} (trained in {svm_train_time:.2f}s)")

    # 6. Select Champion Model
    # Compare F1 score
    if svm_f1 >= lr_f1:
        best_name = "Calibrated Support Vector Machine (LinearSVC)"
        best_model = svm_model
        best_acc = svm_acc
        best_f1 = svm_f1
    else:
        best_name = "Logistic Regression"
        best_model = lr_model
        best_acc = lr_acc
        best_f1 = lr_f1

    print(f"\n[Stage 5/5] Best Model Selected: {best_name}")
    print(f" -> Accuracy: {best_acc * 100:.2f}% | F1-Score: {best_f1:.4f}")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "fake_news_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_size": len(df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "selected_model": best_name,
        "models": {
            "Logistic Regression": {
                "accuracy": round(float(lr_acc) * 100, 2),
                "f1_score": round(float(lr_f1), 4),
                "precision": round(float(lr_prec), 4),
                "recall": round(float(lr_rec), 4),
                "training_time_sec": round(lr_train_time, 2),
                "confusion_matrix": lr_cm,
            },
            "Support Vector Machine": {
                "accuracy": round(float(svm_acc) * 100, 2),
                "f1_score": round(float(svm_f1), 4),
                "precision": round(float(svm_prec), 4),
                "recall": round(float(svm_rec), 4),
                "training_time_sec": round(svm_train_time, 2),
                "confusion_matrix": svm_cm,
            }
        },
        "best_metrics": {
            "accuracy": round(float(best_acc) * 100, 2),
            "f1_score": round(float(best_f1), 4),
        }
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    total_duration = time.time() - start_total_time
    print(f"\nArtifacts successfully saved:")
    print(f"  - Model:      {model_path}")
    print(f"  - Vectorizer: {vectorizer_path}")
    print(f"  - Metrics:    {metrics_path}")
    print(f"\nTraining pipeline completed successfully in {total_duration:.2f} seconds!")
    return metrics_payload


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"\nError during model training: {e}", file=sys.stderr)
        sys.exit(1)
