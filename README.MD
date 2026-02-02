<!-- https://www.kaggle.com/datasets/noahx1/csgo-steam-reviews?select=csgo_steam_reviews.csv -->

# Analiza Sentymentu Recenzji Steam (CS:GO) 🎮

Projekt polega na analizie danych tekstowych i metadanych z recenzji gry Counter-Strike: Global Offensive. System klasyfikuje opinie jako pozytywne lub negatywne oraz analizuje korelacje między statystykami graczy a ich zadowoleniem.

---

## DANE
https://www.kaggle.com/datasets/noahx1/csgo-steam-reviews?select=csgo_steam_reviews.csv

## 🛠️ Wykorzystane Biblioteki

### Analiza i Przetwarzanie Danych
* **Pandas & NumPy**: Manipulacja danymi i operacje macierzowe.
* **SQLite3**: Obsługa relacyjnej bazy danych.

### Machine Learning (Scikit-Learn)
* `TfidfVectorizer`: Transformacja tekstu na wektory liczbowe.
* **Modele Klasyfikacyjne**: 
    * `MultinomialNB` & `BernoulliNB` (Naiwny Bayes)
    * `KNeighborsClassifier` (k-NN)
    * `LinearSVC` (SVM)
    * `RandomForestClassifier` & `GradientBoostingClassifier` (Modele zespołowe)
* **Ewaluacja**: `classification_report`, `confusion_matrix`, `accuracy_score`.

### Wizualizacja i NLP
* **Matplotlib & Seaborn**: Wykresy rozkładu i korelacji.
* **WordCloud**: Generowanie chmur słów dla najczęstszych fraz.
* **RE (Regex)**: Zaawansowane czyszczenie tekstu z tagów i znaków specjalnych.

---

## ✅ Spełnione Kryteria Projektowe

### 1. Wielkość i Przechowywanie Danych (18 pkt)
* **Rozmiar**: Zbiór zawiera **15 586 rekordów** (Kryterium 10k-25k – **8 pkt**).
* **Baza Danych**: Wykorzystano relacyjną bazę **SQLite** do przechowywania danych (Kryterium – **10 pkt**).

### 2. Klasyfikacja Dokumentów (30 pkt)
Zaimplementowano cztery grupy modeli klasyfikacyjnych (**30 pkt**):
* **Naiwny Bayes**: Zarówno model Multinomial, jak i Bernoulli.
* **k-Najbliższych Sąsiadów**: `KNeighborsClassifier`.
* **SVM**: Maszyna wektorów nośnych (`LinearSVC`).
* **Modele Zespołowe**: Las Losowy oraz Gradient Boosting.

### 3. Analiza Tekstu i Sentymentu (20 pkt)
* **Analiza Sentymentu**: Klasyfikacja na podstawie kolumny `voted_up` (**10 pkt**).
* **Analiza Statystyczna**: Obliczanie długości dokumentów, liczby słów oraz średnich dla grup (**10 pkt**).
* **Czyszczenie**: Usuwanie szumów informacyjnych (tagi Steam, znaki interpunkcyjne).

### 4. Filtrowanie i Metadane (10 pkt)
Skrypt umożliwia filtrowanie danych wejściowych (Kryterium – **1 pkt za filtr**):
1. Język recenzji.
2. Minimalny czas gry (`playtime_hours`).
3. Status zakupu na Steam.
4. Liczba głosów "przydatne".
5. Liczba posiadanych gier przez autora.

### 5. Wizualizacje (16 pkt)
Skrypt zawiera 4 wizualizacje
---

## 📈 Wyniki Analizy
Najlepszą skuteczność osiągnął model **LinearSVC**, uzyskując dokładność na poziomie **90.76%**. Analiza korelacji wykazała, że długość recenzji jest słabo skorelowana z sentymentem, co sugeruje, że zarówno krótkie, jak i długie recenzje mogą być równie emocjonalne.

---

