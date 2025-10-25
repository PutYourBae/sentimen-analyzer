# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from preprocess import preprocess_series

DATA_PATH = "../data/reviews.csv"
MODEL_PATH = "../models/sentiment_model.joblib"

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan.")
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV harus punya kolom 'text' dan 'label'.")
    return df

def train():
    print("📘 Memuat data...")
    df = load_data()
    X, sarcasm_flags = preprocess_series(df["text"])
    y = df["label"]

    # Optional: tambahkan fitur sarcasm sebagai boolean (dalam kolom baru) - kita gabungkan ke teks
    X_with_meta = X + " " + sarcasm_flags.map(lambda f: "sarkasme" if f else "")

    print("📗 Membagi data latih & uji...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_meta, y, test_size=0.2, random_state=42, stratify=y
    )

    print("📙 Membuat pipeline model (TF-IDF + MultinomialNB)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=8000)),
        ("clf", MultinomialNB())
    ])

    print("📕 Melatih model...")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("\n🎯 Akurasi:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model disimpan ke: {MODEL_PATH}")

if __name__ == "__main__":
    train()
