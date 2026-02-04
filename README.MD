# 🎮 Analiza Sentymentu Recenzji Steam (CS:GO / CS2)

Projekt stanowi kompleksową platformę analityczną do badania opinii graczy Counter-Strike 2. System integruje dane historyczne z bazy SQL z danymi pobieranymi w czasie rzeczywistym przez API Steam, oferując zaawansowane filtrowanie, przetwarzanie języka naturalnego (NLP) oraz automatyczną klasyfikację sentymentu za pomocą wielu modeli uczenia maszynowego.

---

## 📂 Dane i Przechowywanie
* **Źródło danych**: [Kaggle CS:GO Steam Reviews](https://www.kaggle.com/datasets/noahx1/csgo-steam-reviews).
* **Wolumen**: Zbiór zawiera **15 586 rekordów**.
* **Baza Danych**: Wykorzystano relacyjną bazę **SQLite** (`reviews.db`) do przechowywania i odpytywania danych.
* **Unikatowe Cechy**: Wykorzystano ponad **13 unikatowych pól** (m.in. `voted_up`, `playtime`, `weighted_vote_score`, `early_access`).

---

## 🛠️ Stack Technologiczny i Biblioteki

### Interfejs i Infrastruktura
* **Streamlit**: Biblioteka wykorzystana do budowy aplikacji webowej.
* **Requests**: Integracja z zewnętrznym **Steam Web API** (pobieranie liczby graczy online).
* **Caching**: Użycie `@st.cache_data` oraz `@st.cache_resource` do optymalizacji wydajności.

### Przetwarzanie Tekstu (NLP)
* **SpaCy**: Zaawansowana analiza morfologiczna i lematyzacja tekstów.
* **Scikit-learn (TF-IDF)**: Nadawanie wag termom w celu wektoryzacji.
* **Regex (RE)**: Czyszczenie tekstu z tagów HTML i znaków specjalnych za pomocą wyrażeń regularnych.

### Klasyfikacja i ML (Scikit-Learn)
Wytrenowano i porównano własne modele klasyfikacyjne:
* **Multinomial Naive Bayes** oraz **Bernoulli Naive Bayes**.
* **k-Nearest Neighbors** (k-NN).
* **Support Vector Machine** (LinearSVC).
* **Random Forest** oraz **Gradient Boosting**.

---

## ✅ Realizacja Kryteriów Projektowe

### Aplikacja Webowa (10/15 pkt)
* **Streamlit**

## Wielkość i przechowywanie analizowanego zbioru danych (18/20 pkt)
* **Baza Danych SQLlite**
* **Ilość danych**: ponad 15000 rekordów

### Klasyfikacja Dokumentów (30/30 pkt)
Zaimplementowano 4 grupy modeli klasyfikacyjnych, spełniając maksymalny limit kategorii. Aplikacja prezentuje ewaluację modeli poprzez miary: **Accuracy, Precision, Recall oraz F1-Score**.

### Analiza Tekstu i Sentymentu (20/40 pkt)
* **Analiza Sentymentu**: Wykrywanie emocji w recenzjach (10 pkt).
* **Analiza Morfologiczna**: Lematyzacja z użyciem SpaCy (10 pkt).

### Wizualizacja Wyników (20/20 pkt)
Wykorzystano 5 unikatowych typów wykresów (limit kategorii osiągnięty):
1. **Wykres Kołowy**: Rozkład sentymentu.
2. **Histogram**: Rozkład czasu gry i długości recenzji.
3. **Wykres Słupkowy**: Porównanie efektywności modeli.
4. **Chmura Słów (WordCloud)**: Najczęstsze termy dla pozytywów/negatywów.
5. **Mapa Cieplna (Heatmap)**: Macierz pomyłek (Confusion Matrix).

### Filtrowanie (4/10 pkt)
* **Filtry (4 pkt)**: 4 interaktywne kontrolek (Czas gry, Upvotes, Liczba recenzji, Liczba gier).

## Dodatkowe funkcje (10 pkt)
* **Cache (5 pkt)**: Zapisywanie wyników w pamięci podręcznej.
* **Zasoby (5 pkt)**: Wykorzystanie Steam API jako dodatkowego, unikatowego źródła danych.

---

## 📈 Wyniki i Wnioski
Najwyższą skuteczność (F1-Score) wykazuje model **LinearSVC**. Aplikacja pozwala na testowanie modelu "na żywo" w zakładce **Predictions**, gdzie użytkownik może wpisać własną opinię, a system dokona jej automatycznej klasyfikacji na podstawie wytrenowanego modelu.

---

**Suma punktów: 98 pkt**
