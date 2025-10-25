# src/app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from preprocess import clean_text, tokenize_and_stem, detect_sarcasm, preprocess_series

MODEL_PATH = "../models/sentiment_model.joblib"
DATA_PATH = "../data/reviews.csv"

app = Flask(__name__, template_folder="../templates")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model belum ada. Jalankan src/train.py dulu untuk membuat model.")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    # load dataset ringkasan untuk chart
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # hitung distribusi label original
        dist = df['label'].value_counts().to_dict()
    else:
        dist = {}
    return render_template("index.html", dist=dist)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get("text", "")
    if not data:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    processed = tokenize_and_stem(clean_text(data))
    sarcasm = detect_sarcasm(data)
    # tambahkan flag sarcasm ke input seperti saat training
    input_text = processed + (" sarkasme" if sarcasm else "")
    pred = model.predict([input_text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = dict(zip(model.classes_, model.predict_proba([input_text])[0].round(3).tolist()))
    return jsonify({"input": data, "processed": input_text, "prediction": pred, "probabilities": proba, "sarcasm": bool(sarcasm)})

@app.route("/stats", methods=["GET"])
def stats():
    """Kirim data ringkasan (counts) untuk chart: dari dataset dan prediksi sample."""
    result = {}
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        result['dataset_counts'] = df['label'].value_counts().to_dict()
    else:
        result['dataset_counts'] = {}

    # contoh: jalankan prediksi pada seluruh dataset (bisa mahal jika besar)
    try:
        df = pd.read_csv(DATA_PATH)
        X, sarcasm = preprocess_series(df['text'])
        X_with_meta = X + " " + sarcasm.map(lambda f: "sarkasme" if f else "")
        preds = model.predict(X_with_meta.tolist())
        result['predicted_counts'] = pd.Series(preds).value_counts().to_dict()
    except Exception as e:
        result['predicted_counts'] = {}
        result['error'] = str(e)

    return jsonify(result)

if __name__ == "__main__":
    print("🚀 Server berjalan di http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
