# nlp-dashboard/src/topic_model.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from src.preprocess import TextPreprocessor

class TopicModeler:
    def __init__(self, n_topics: int = 4, n_words: int = 7):
        self.n_topics = n_topics
        self.n_words = n_words
        self.preprocessor = TextPreprocessor()
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        self.lda_model = LDA(n_components=n_topics, random_state=42)

    def fit_extract_topics(self, text_list: list) -> dict:
        """
        Processes text lists, runs unsupervised LDA, and clusters key thematic groups.
        """
        cleaned_corpus = [self.preprocessor.process_for_modeling(text) for text in text_list]
        
        # Guard against completely empty or tiny datasets
        if len(cleaned_corpus) < self.n_topics or not any(cleaned_corpus):
            return {f"Topic #{i+1}": ["Insufficient data or words to model topics"] for i in range(self.n_topics)}
            
        try:
            dtm = self.vectorizer.fit_transform(cleaned_corpus)
            self.lda_model.fit(dtm)
            
            words = self.vectorizer.get_feature_names_out()
            topics_dict = {}
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_word_indices = topic.argsort()[:-self.n_words - 1:-1]
                top_words = [words[i] for i in top_word_indices]
                topics_dict[f"Topic #{topic_idx + 1}"] = top_words
                
            return topics_dict
        except Exception:
            return {f"Topic #{i+1}": ["Processing error"] for i in range(self.n_topics)}