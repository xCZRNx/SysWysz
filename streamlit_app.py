# streamlit_app.py
import streamlit as st
import sqlite3
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import spacy
import subprocess
import sys
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="CS:GO Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive-text {
        color: #2ca02c;
        font-weight: bold;
    }
    .negative-text {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD SPACY MODEL =====
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.warning("Downloading Spacy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load('en_core_web_sm')
    return nlp

# ===== TEXT PROCESSING FUNCTIONS =====
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@st.cache_resource
def get_nlp():
    return load_spacy_model()

def lemmatize_text(text, nlp):
    if not text:
        return ""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(lemmas)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('reviews.db')
        query = """
        SELECT 
            review, 
            voted_up,
            written_during_early_access,
            author_playtime_forever, 
            author_playtime_last_two_weeks,
            author_last_played,
            author_playtime_at_review,
            author_num_reviews, 
            "author.num_games_owned", 
            votes_up, 
            language
        FROM reviews
        """
        df = pd.read_sql(query, conn)
        df['author_playtime_hours'] = df['author_playtime_forever'] / 60
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ===== FETCH CONCURRENT PLAYERS FROM STEAM API =====
@st.cache_data(ttl=3600)
def get_concurrent_players():
    try:
        url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=730"
        response = requests.get(url, timeout=10).json()
        player_count = response['response']['player_count']
        return player_count
    except Exception as e:
        st.warning(f"Could not fetch concurrent players: {str(e)}")
        return None

# ===== MAIN APP =====
def main():
    st.title("🎮 CS:GO Steam Reviews Sentiment Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar - Filtering
    st.sidebar.header("⚙️ Filter Parameters")
    
    min_godzin = st.sidebar.slider("Minimum playtime (hours)", 0, 1000, 50)
    min_recenzji = st.sidebar.slider("Minimum reviews by author", 1, 100, 1)
    min_votes_up = st.sidebar.slider("Minimum upvotes", 0, 100, 1)
    posiadane_gry = st.sidebar.slider("Minimum games owned", 1, 1000, 1)
    
    # Filter data
    df_filtered = df[
        (df['author_playtime_hours'] >= min_godzin) &
        (df['author_num_reviews'] >= min_recenzji) &
        (df['votes_up'] >= min_votes_up) &
        (df['author.num_games_owned'] >= posiadane_gry)
    ].copy()
    
    # Display statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Statistics")
    st.sidebar.metric("Original Records", len(df))
    st.sidebar.metric("Filtered Records", len(df_filtered))
    st.sidebar.metric("Avg Playtime (h)", f"{df['author_playtime_hours'].mean():.1f}")
    st.sidebar.metric("Avg Reviews", f"{df['author_num_reviews'].mean():.1f}")
    
    # TAB 1: DATA OVERVIEW
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔤 Text Analysis",
        "🤖 Model Training",
        "📈 Results",
        "💬 Predictions"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        # Get concurrent players from SteamDB
        concurrent_players = get_concurrent_players()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if concurrent_players:
                st.metric("🎮 Concurrent Players", f"{concurrent_players:,}")
            else:
                st.metric("🎮 Concurrent Players", "N/A")
        
        with col2:
            positive_count = (df_filtered['voted_up'] == 1).sum()
            st.metric("Positive Reviews", positive_count)
        
        with col3:
            negative_count = (df_filtered['voted_up'] == 0).sum()
            st.metric("Negative Reviews", negative_count)
        
        with col4:
            positive_pct = (positive_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with col5:
            avg_votes = df_filtered['votes_up'].mean()
            st.metric("Avg Upvotes", f"{avg_votes:.1f}")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sentiment_counts = df_filtered['voted_up'].value_counts().sort_index()
            colors = ['#d62728', '#2ca02c']
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=['Negative', 'Positive'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            ax.set_title(f'Sentiment Distribution (n={len(df_filtered)})', fontweight='bold', fontsize=12)
            st.pyplot(fig)
        
    
        # Sample reviews
        st.subheader("Sample Reviews")
        review_type = st.radio("Show:", ["Positive Reviews", "Negative Reviews"], horizontal=True)
        sentiment_value = 1 if review_type == "Positive Reviews" else 0
        
        sample_reviews = df_filtered[df_filtered['voted_up'] == sentiment_value].sample(
            min(5, len(df_filtered[df_filtered['voted_up'] == sentiment_value]))
        )
        
        for idx, row in sample_reviews.iterrows():
            with st.expander(f"Review #{idx} - Upvotes: {row['votes_up']}"):
                st.write(row['review'])
    
    with tab2:
        st.subheader("Text Processing & Analysis")
        
        # Process text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Cleaning reviews...")
        df_filtered['clean_review'] = df_filtered['review'].apply(clean_text)
        progress_bar.progress(33)
        
        status_text.text("Calculating statistics...")
        df_filtered['review_length'] = df_filtered['clean_review'].str.len()
        df_filtered['word_count'] = df_filtered['clean_review'].str.split().str.len()
        progress_bar.progress(66)
        
        status_text.text("Lemmatizing text...")
        nlp = get_nlp()
        df_filtered['lemmatized_review'] = df_filtered['clean_review'].apply(lambda x: lemmatize_text(x, nlp))
        progress_bar.progress(100)
        status_text.text("✅ Text processing complete!")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Review Length", f"{df_filtered['review_length'].mean():.0f} chars")
        with col2:
            st.metric("Avg Word Count", f"{df_filtered['word_count'].mean():.0f} words")
        with col3:
            st.metric("Max Review Length", f"{df_filtered['review_length'].max():.0f} chars")
        with col4:
            st.metric("Min Review Length", f"{df_filtered['review_length'].min():.0f} chars")
        
        # Word count distribution
        st.subheader("Review Length Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_filtered[df_filtered['voted_up'] == 1]['word_count'], bins=50, alpha=0.7, label='Positive', color='#2ca02c')
            ax.hist(df_filtered[df_filtered['voted_up'] == 0]['word_count'], bins=50, alpha=0.7, label='Negative', color='#d62728')
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Word Count Distribution by Sentiment')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_filtered[df_filtered['voted_up'] == 1]['review_length'], bins=50, alpha=0.7, label='Positive', color='#2ca02c')
            ax.hist(df_filtered[df_filtered['voted_up'] == 0]['review_length'], bins=50, alpha=0.7, label='Negative', color='#d62728')
            ax.set_xlabel('Character Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Character Count Distribution by Sentiment')
            ax.legend()
            st.pyplot(fig)
        
        # Word clouds
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Reviews**")
            positive_text = " ".join(df_filtered[df_filtered['voted_up'] == 1]['clean_review'].sample(
                min(500, len(df_filtered[df_filtered['voted_up'] == 1]))
            ).values)
            if positive_text:
                wordcloud_pos = WordCloud(width=400, height=400, background_color='white', colormap='Greens').generate(positive_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.write("**Negative Reviews**")
            negative_text = " ".join(df_filtered[df_filtered['voted_up'] == 0]['clean_review'].sample(
                min(500, len(df_filtered[df_filtered['voted_up'] == 0]))
            ).values)
            if negative_text:
                wordcloud_neg = WordCloud(width=400, height=400, background_color='white', colormap='Reds').generate(negative_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    
    with tab3:
        st.subheader("🤖 Model Training & Evaluation")
        
        if 'training_complete' not in st.session_state:
            if st.button("🚀 Start Training Models", key="train_button"):
                with st.spinner("Training models... This may take a few minutes"):
                    # TF-IDF Vectorization
                    st.write("Vectorizing text with TF-IDF...")
                    tfidf = TfidfVectorizer(max_features=2000, stop_words='english', min_df=5, max_df=0.8)
                    X = tfidf.fit_transform(df_filtered['lemmatized_review'])
                    y = df_filtered['voted_up']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.tfidf = tfidf
                    st.session_state.feature_names = np.array(tfidf.get_feature_names_out())
                    
                    results = {}
                    
                    # 1. Multinomial NB
                    st.write("Training Multinomial Naive Bayes...")
                    model = MultinomialNB(alpha=1.0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                    results['Multinomial NB'] = {
                        'model': model,
                        'pred': y_pred,
                        'acc': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    }
                    
                    # 2. Bernoulli NB
                    st.write("Training Bernoulli Naive Bayes...")
                    ber_model = BernoulliNB(alpha=1.0)
                    ber_model.fit(X_train, y_train)
                    y_pred_ber = ber_model.predict(X_test)
                    acc_ber = accuracy_score(y_test, y_pred_ber)
                    precision_ber, recall_ber, f1_ber, _ = precision_recall_fscore_support(y_test, y_pred_ber, average='weighted')
                    results['Bernoulli NB'] = {
                        'model': ber_model,
                        'pred': y_pred_ber,
                        'acc': acc_ber,
                        'precision': precision_ber,
                        'recall': recall_ber,
                        'f1': f1_ber,
                    }
                    
                    # 3. LinearSVC
                    st.write("Training LinearSVC...")
                    svm_model = LinearSVC(random_state=42, max_iter=2000, C=1.0)
                    svm_model.fit(X_train, y_train)
                    y_pred_svm = svm_model.predict(X_test)
                    acc_svm = accuracy_score(y_test, y_pred_svm)
                    precision_svm, recall_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')
                    results['LinearSVC'] = {
                        'model': svm_model,
                        'pred': y_pred_svm,
                        'acc': acc_svm,
                        'precision': precision_svm,
                        'recall': recall_svm,
                        'f1': f1_svm
                    }
                    
                    # 4. k-NN
                    st.write("Training k-Nearest Neighbors...")
                    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                    knn_model.fit(X_train, y_train)
                    y_pred_knn = knn_model.predict(X_test)
                    acc_knn = accuracy_score(y_test, y_pred_knn)
                    precision_knn, recall_knn, f1_knn, _ = precision_recall_fscore_support(y_test, y_pred_knn, average='weighted')
                    results['k-NN (k=5)'] = {
                        'model': knn_model,
                        'pred': y_pred_knn,
                        'acc': acc_knn,
                        'precision': precision_knn,
                        'recall': recall_knn,
                        'f1': f1_knn,
                    }
                    
                    # 5. Random Forest
                    st.write("Training Random Forest...")
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
                    rf_model.fit(X_train, y_train)
                    y_pred_rf = rf_model.predict(X_test)
                    acc_rf = accuracy_score(y_test, y_pred_rf)
                    precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
                    results['Random Forest'] = {
                        'model': rf_model,
                        'pred': y_pred_rf,
                        'acc': acc_rf,
                        'precision': precision_rf,
                        'recall': recall_rf,
                        'f1': f1_rf,
                    }
                    
                    # 6. Gradient Boosting
                    st.write("Training Gradient Boosting...")
                    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)
                    gb_model.fit(X_train, y_train)
                    y_pred_gb = gb_model.predict(X_test)
                    acc_gb = accuracy_score(y_test, y_pred_gb)
                    precision_gb, recall_gb, f1_gb, _ = precision_recall_fscore_support(y_test, y_pred_gb, average='weighted')
                    results['Gradient Boosting'] = {
                        'model': gb_model,
                        'pred': y_pred_gb,
                        'acc': acc_gb,
                        'precision': precision_gb,
                        'recall': recall_gb,
                        'f1': f1_gb,
                    }
                    
                    st.session_state.results = results
                    st.session_state.training_complete = True
                    st.success("✅ Training complete!")
                    st.rerun()
        
        else:
            st.success("✅ Models already trained!")
            results_df = pd.DataFrame([
                {'Model': k, 'Accuracy': v['acc'], 'Precision': v['precision'], 
                 'Recall': v['recall'], 'F1-Score': v['f1']}
                for k, v in st.session_state.results.items()
            ])
            st.dataframe(results_df, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Results & Visualizations")
        
        if 'training_complete' in st.session_state and st.session_state.training_complete:
            results = st.session_state.results
            results_df = pd.DataFrame([
                {'Model': k, 'Accuracy': v['acc'], 'Precision': v['precision'], 
                 'Recall': v['recall'], 'F1-Score': v['f1']}
                for k, v in results.items()
            ])
            
            # Model comparison
            st.subheader("Model Performance Comparison")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
                bars = ax.barh(results_df['Model'], results_df[metric], color=colors)
                ax.set_xlabel(metric, fontsize=11, fontweight='bold')
                ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                ax.set_xlim([0, 1])
                
                for bar, val in zip(bars, results_df[metric]):
                    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                            va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model
            best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
            best_f1 = results_df['F1-Score'].max()
            st.info(f"🏆 Best Model: **{best_model_name}** with F1-Score: **{best_f1:.4f}**")
            
            # Confusion matrices
            st.subheader("Confusion Matrices (Top 4 Models)")
            top_models = results_df.nlargest(4, 'F1-Score')['Model'].tolist()
            fig, axes = plt.subplots(2, 2, figsize=(12, 11))
            axes = axes.flatten()
            
            for idx, model_name in enumerate(top_models):
                y_pred_model = results[model_name]['pred']
                y_test = st.session_state.y_test
                cm = confusion_matrix(y_test, y_pred_model)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'])
                
                f1_val = results[model_name]['f1']
                acc_val = results[model_name]['acc']
                axes[idx].set_title(f'{model_name}\nF1={f1_val:.3f} | Acc={acc_val:.3f}', fontweight='bold')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # TF-IDF Analysis
            st.subheader("Top 15 TF-IDF Terms")
            
            X = st.session_state.X_test  # or X_train
            y_test = st.session_state.y_test
            feature_names = st.session_state.feature_names
            
            # Positive
            positive_indices = (y_test == 1).values
            positive_X = X[positive_indices]
            positive_mean = np.asarray(positive_X.mean(axis=0)).ravel()
            top_positive_idx = positive_mean.argsort()[-15:][::-1]
            
            # Negative
            negative_indices = (y_test == 0).values
            negative_X = X[negative_indices]
            negative_mean = np.asarray(negative_X.mean(axis=0)).ravel()
            top_negative_idx = negative_mean.argsort()[-15:][::-1]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].barh(feature_names[top_positive_idx], positive_mean[top_positive_idx], color='#2ca02c', alpha=0.8)
            axes[0].set_xlabel('Mean TF-IDF Weight', fontweight='bold')
            axes[0].set_title('Top 15 Terms in POSITIVE Reviews', fontsize=12, fontweight='bold')
            axes[0].invert_yaxis()
            
            axes[1].barh(feature_names[top_negative_idx], negative_mean[top_negative_idx], color='#d62728', alpha=0.8)
            axes[1].set_xlabel('Mean TF-IDF Weight', fontweight='bold')
            axes[1].set_title('Top 15 Terms in NEGATIVE Reviews', fontsize=12, fontweight='bold')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.warning("Please train models first in the 'Model Training' tab")
    
    with tab5:
        st.subheader("💬 Predict Review Sentiment")
        
        if 'training_complete' in st.session_state and st.session_state.training_complete:
            user_review = st.text_area("Enter a review:", height=150, placeholder="Type your CS:GO review here...")
            
            if st.button("🔮 Predict Sentiment"):
                if user_review.strip():
                    # Preprocess
                    cleaned = clean_text(user_review)
                    nlp = get_nlp()
                    lemmatized = lemmatize_text(cleaned, nlp)
                    
                    # Vectorize
                    tfidf = st.session_state.tfidf
                    X_user = tfidf.transform([lemmatized])
                    
                    # Predict with best model
                    results = st.session_state.results
                    results_df = pd.DataFrame([
                        {'Model': k, 'F1-Score': v['f1']}
                        for k, v in results.items()
                    ])
                    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
                    best_model = results[best_model_name]['model']
                    
                    prediction = best_model.predict(X_user)[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.success("✅ **POSITIVE REVIEW**")
                            st.balloons()
                        else:
                            st.error("❌ **NEGATIVE REVIEW**")
                    
                    with col2:
                        st.info(f"Model used: **{best_model_name}**")
                else:
                    st.warning("Please enter a review")
        else:
            st.warning("Please train models first in the 'Model Training' tab")

if __name__ == "__main__":
    main()