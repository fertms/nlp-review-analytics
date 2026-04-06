# Pré-processamento de texto para análise NLP
import pandas as pd
import numpy as np
import re
import nltk
from pathlib import Path

# Download dos recursos NLTK necessários
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

RAW       = Path("../data/raw")
PROCESSED = Path("../data/processed")
PROCESSED.mkdir(exist_ok=True)

def clean_text(text: str) -> str:
    # Converte para minúsculas
    text = text.lower()
    # Remove caracteres especiais e números
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text: str, stop_words: set, lemmatizer) -> str:
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    stop_words  = set(stopwords.words("english"))
    lemmatizer  = WordNetLemmatizer()

    df["clean_text"]     = df["review_text"].apply(clean_text)
    df["processed_text"] = df["clean_text"].apply(
        lambda x: tokenize_and_lemmatize(x, stop_words, lemmatizer)
    )
    df["text_length"]    = df["review_text"].apply(len)
    df["word_count"]     = df["review_text"].apply(lambda x: len(x.split()))
    df["review_date"]    = pd.to_datetime(df["review_date"])
    df["review_month"]   = df["review_date"].dt.month
    df["review_year"]    = df["review_date"].dt.year
    return df

if __name__ == "__main__":
    df = pd.read_csv(RAW / "reviews.csv")
    df = preprocess(df)
    df.to_csv(PROCESSED / "reviews_processed.csv", index=False, encoding="utf-8-sig")
    print(f"Preprocessamento concluído: {len(df)} reviews")
    print(df[["review_text", "clean_text", "processed_text", "word_count"]].head(3))