# nlp-dashboard/src/sentiment.py
import numpy as np
from src.utils import load_model
from src.preprocess import TextPreprocessor

class SentimentAnalyzer:
    def __init__(self, model_path: str = "models/sentiment_model.pkl"):
        self.preprocessor = TextPreprocessor()
        # Attempt to load a trained model pipeline (e.g., TF-IDF + LogisticRegression)
        self.model = load_model(model_path)
        
    def _rule_based_fallback(self, text: str) -> dict:
        """
        A lightweight fallback analyzer used when no ML model is trained.
        Uses a clean keyword matching lexicon for speed and reliability.
        """
        tokens = self.preprocessor.tokenize_and_remove_stopwords(text)
        if not tokens:
            return {"label": "Neutral", "score": 0.5}
            
        # Hardcoded semantic weights for fallback execution
        pos_words = {'love', 'loved', 'excellent', 'fantastic', 'great', 'amazing', 'perfect', 'beautiful', 'efficient', 'good', 'smooth', 'value'}
        neg_words = {'terrible', 'disappointed', 'waste', 'poor', 'cheap', 'fragile', 'bad', 'broken', 'slow', 'glitchy', 'crashing', 'avoid', 'uncomfortable'}
        
        pos_count = sum(1 for w in tokens if w in pos_words)
        neg_count = sum(1 for w in tokens if w in neg_words)
        
        total = pos_count + neg_count
        if total == 0:
            return {"label": "Neutral", "score": 0.5}
            
        pos_ratio = pos_count / total
        if pos_ratio > 0.55:
            return {"label": "Positive", "score": float(np.round(0.6 + (pos_ratio * 0.4), 2))}
        elif pos_ratio < 0.45:
            return {"label": "Negative", "score": float(np.round(0.6 + ((1 - pos_ratio) * 0.4), 2))}
        else:
            return {"label": "Neutral", "score": 0.5}

    def analyze_text(self, text: str) -> dict:
        """
        Analyzes a single string input. Returns a dictionary containing
        the predicted structural label and confidence score.
        """
        if not text or not text.strip():
            return {"label": "Neutral", "score": 0.0}
            
        if self.model is not None:
            try:
                # Expects a pipeline containing vectorizer + classifier
                prediction = self.model.predict([text])[0]
                probabilities = self.model.predict_proba([text])[0]
                max_idx = np.argmax(probabilities)
                
                # Maps model output to labels
                labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                label = labels_map.get(prediction, str(prediction))
                score = float(probabilities[max_idx])
                
                return {"label": label, "score": round(score, 2)}
            except Exception:
                # Graceful degradation on model dimension mismatch or failure
                return self._rule_based_fallback(text)
        else:
            return self._rule_based_fallback(text)

    def analyze_dataframe(self, df, text_column: str):
        """
        Applies analysis context vectorially over an ingested pandas dataframe.
        Appends pipeline-safe columns 'sentiment_label' and 'sentiment_score'.
        """
        results = df[text_column].apply(self.analyze_text)
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        return df