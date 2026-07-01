# nlp-dashboard/src/preprocess.py
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Point NLTK at the bundled data folder shipped with the repo instead of
# downloading resources at runtime. Assumes this file lives at
# nlp-dashboard/src/preprocess.py and data lives at nlp-dashboard/nltk_data/
# (generate it once locally with download_nltk_data.py, then commit it).
_NLTK_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "nltk_data"
)
if _NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(_NLTK_DATA_PATH)

# Fail fast and clearly if the bundled data is missing, rather than
# silently falling back to a slow/flaky network download.
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    raise RuntimeError(
        "Bundled NLTK data not found at "
        f"'{_NLTK_DATA_PATH}'. Run `python download_nltk_data.py` locally "
        "and commit the resulting nltk_data/ folder to the repo."
    ) from e


class TextPreprocessor:
    def __init__(self, lower=True, remove_punctuation=True, remove_digits=True):
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        """Applies basic pipeline cleansers: lowers, strips tags, and drops symbols."""
        if not isinstance(text, str):
            return ""

        if self.lower:
            text = text.lower()

        # Strip HTML/XML tags if any
        text = re.sub(r'<[^>]+>', ' ', text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        if self.remove_digits:
            text = re.sub(r'\d+', '', text)

        # Condense spacing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_remove_stopwords(self, text: str) -> list:
        """Cleans input, tokenizes, and strips standard English stopwords."""
        cleaned_text = self.clean(text)
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        return filtered_tokens

    def process_for_modeling(self, text: str) -> str:
        """Returns a single clean string suitable for TF-IDF / Vectorizers."""
        return " ".join(self.tokenize_and_remove_stopwords(text))
