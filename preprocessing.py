import re
import nltk
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_punctuation(text):
    if isinstance(text, str):
        # Hapus simbol-simbol yang tidak diinginkan
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower()  # Normalisasi huruf menjadi huruf kecil
    else:
        return str(text)  # Ubah ke string jika bukan string

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word.lower() not in stop_words]

def preprocess_text(text):
    if isinstance(text, str):
        # Ubah empty string menjadi NaN value
        if text.strip() == '':
            return np.nan
        text = remove_punctuation(text)
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        return ' '.join(tokens)
    else:
        return str(text)

def remove_missing_values(df):
    # Hapus baris dengan nilai-nilai yang hilang (NaN)
    df.dropna(inplace=True)
    return df

