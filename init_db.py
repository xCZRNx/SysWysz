import pandas as pd
import sqlite3
import os

csv_file = 'csgo_steam_reviews.csv'
df = pd.read_csv(csv_file)

print(f"📄 Wczytano: {len(df)} rekordów.")


# Usuwanie pustych recenzje
df = df.dropna(subset=['review'])

df['voted_up'] = df['voted_up'].astype(int) 
df['steam_purchase'] = df['steam_purchase'].astype(int)

# Tworzenie bazy danych SQLite
db_name = "reviews.db"
if os.path.exists(db_name):
    os.remove(db_name)

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Zapis DataFrame do SQL
df.to_sql('reviews', conn, if_exists='replace', index=False)

print(f" Baza danych '{db_name}' została utworzona.")
print(f"Zapisano {len(df)} recenzji do tabeli 'reviews'.")

# Testowe zapytanie SQL
cursor.execute("SELECT created, voted_up, review FROM reviews LIMIT 3")
for row in cursor.fetchall():
    verdict = "POZYTYWNA" if row[1] else "NEGATYWNA"
    print(f"[{row[0]}] {verdict}: {row[2][:50]}...")

conn.close()