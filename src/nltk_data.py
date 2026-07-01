# nlp-dashboard/download_nltk_data.py
#
# Run this ONCE locally (not on Streamlit Cloud) from the project root:
#     python download_nltk_data.py
#
# It downloads the 'stopwords' and 'punkt' NLTK resources into a local
# ./nltk_data folder. Commit that folder to your repo so the deployed
# app never needs to reach out to NLTK's servers at runtime.

import os
import nltk

# Folder lives at the project root, alongside app.py
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

print(f"Downloading NLTK resources into: {NLTK_DATA_DIR}")
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)

print("Done. Commit the nltk_data/ folder to your repo, e.g.:")
print("    git add nltk_data/")
print("    git commit -m 'Bundle NLTK data to avoid runtime downloads'")
