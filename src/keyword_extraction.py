# nlp-dashboard/src/keyword_extraction.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import TextPreprocessor

class KeywordExtractor:
    def __init__(self, max_features: int = 20):
        self.max_features = max_features
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=1)

    def extract_top_keywords(self, text_list: list, top_n: int = 10) -> list:
        """
        Extracts structural, high-importance keywords using raw corpus weights (TF-IDF).
        Returns a list of tuples: (word, importance_score)
        """
        cleaned_corpus = [self.preprocessor.process_for_modeling(text) for text in text_list if isinstance(text, str) and text.strip()]
        
        if not cleaned_corpus:
            return []
            
        try:
            tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Aggregate tf-idf scores across entire corpus subset
            sums = tfidf_matrix.sum(axis=0).A1
            data = list(zip(feature_names, sums))
            
            # Sort descending by collective weights
            sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
            return sorted_data[:top_n]
        except Exception:
            return []