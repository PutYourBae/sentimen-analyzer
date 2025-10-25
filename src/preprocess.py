# src/preprocess.py
import re
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan NLTK data tersedia di runtime
try:
    STOPWORDS = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    STOPWORDS = set(stopwords.words('indonesian'))

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Emoticon mapping sederhana ke token kata
EMOTICON_MAP = {
    ":)": "senang",
    ":-)": "senang",
    ":(": "sedih",
    ":-(": "sedih",
    ":D": "senang",
    ":-D": "senang",
    ";)": "nakal",
    ":P": "nakal",
    ":-P": "nakal",
    ":/": "bingung",
    ":'(": "menangis"
}

# Heuristik kata positif/negatif untuk deteksi sarkasme sederhana (bisa dikembangkan)
POSITIVE_WORDS = {"baik", "bagus", "hebat", "terbaik", "senang", "mantap", "cepat", "akur"}
NEGATIVE_WORDS = {"buruk", "jelek", "salah", "error", "lambat", "kecewa", "bug"}

def replace_emoticons(text):
    for emo, token in EMOTICON_MAP.items():
        text = text.replace(emo, f" {token} ")
    return text

def demojize_to_words(text):
    # ubah emoji menjadi :smile: lalu jadi kata 'smile'
    dem = emoji.demojize(text)
    # convert :face_with_tears_of_joy: -> face with tears of joy
    dem = dem.replace(":", " ").replace("_", " ")
    return dem

def detect_sarcasm(text):
    """
    Heuristik sederhana:
    - Jika ada kata tawa 'haha','lol' diikuti/bersama kata negative -> kemungkinan sarkas
    - Jika ada kutipan tanda ' " ' di sekitar kata positif -> kemungkinan sarkas
    """
    t = text.lower()
    if re.search(r"\b(haha|lol|lel|wkwk|wk)\b", t):
        # kalau juga ada kata negatif -> flag sarcasm
        if any(w in t for w in NEGATIVE_WORDS):
            return True
    # tanda kutip di sekitar kata positif, misal: "hebat"
    if re.search(r'["\']\s*\w+\s*["\']', text):
        inner = re.findall(r'["\']\s*(\w+)\s*["\']', text)
        for w in inner:
            if w.lower() in POSITIVE_WORDS:
                return True
    return False

def clean_text(text):
    """1) Lowercase, 2) handle emoticon, emoji, 3) remove urls & non alpha-numeric (tetap simpan spasi)"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = demojize_to_words(text)
    text = replace_emoticons(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    # keep Indonesian letters and numbers and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # Stemming dengan sastrawi
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)

def preprocess_series(series):
    """Proses seluruh kolom teks (pandas Series). Juga kembalikan flag sarcasm per baris."""
    texts = series.fillna("").astype(str)
    clean = texts.map(clean_text)
    sarcasm_flags = clean.map(detect_sarcasm)
    tokenized = clean.map(tokenize_and_stem)
    return tokenized, sarcasm_flags
