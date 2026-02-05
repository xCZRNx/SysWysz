import pandas as pd
import sqlite3
import os
import re
import spacy
import subprocess
import sys

csv_file = 'csgo_steam_reviews.csv'
df = pd.read_csv(csv_file)

print(f"Wczytano: {len(df)} rekordów.")

df = df.dropna(subset=['review'])

df['voted_up'] = df['voted_up'].astype(int) 
df['steam_purchase'] = df['steam_purchase'].astype(int)

def clean_text(text):
    """Clean and normalize review text"""
    if text is None:
        return ""
    text = str(text).lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("   Pobieranie modelu spacy...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    """Lemmatize text using spacy"""
    if not text:
        return ""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(lemmas)

# ===== TEXT PROCESSING =====
df['clean_review'] = df['review'].apply(clean_text)

df['lemmatized_review'] = df['clean_review'].apply(lemmatize_text)


# Tworzenie bazy danych SQLite
db_name = "reviews.db"
if os.path.exists(db_name):
    os.remove(db_name)

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Zapis DataFrame do SQL
df.to_sql('reviews', conn, if_exists='replace', index=False)

print(f"   Zapisano {len(df)} recenzji do tabeli 'reviews'.")

conn.close()