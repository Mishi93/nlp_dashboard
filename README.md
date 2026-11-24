# Customer Sentiment & Text Analytics Dashboard

A Streamlit dashboard that analyses customer reviews: sentiment per review, TF-IDF keywords, LDA topics, and category and time breakdowns — all from a single CSV file.

**Live demo:** none yet — no demo link is listed in the repo. Run it locally (see below).

> This is a demonstration project built on a 320-row sample file, not a production tool. It has no file upload in the app, no tests and no deployment — see [Limitations](#limitations) for exactly what is and is not there.

## What it does

1. Loads the bundled reviews from `data/sample_reviews.csv`.
2. Scores every review as Negative, Neutral or Positive, using the saved model in `models/sentiment_model.pkl` (with a rule-based fallback if the model cannot be used).
3. Lets you filter by product category and by date range in the sidebar.
4. Shows four KPIs, four charts, top keywords, four LDA topics and the full review table.

## Run it locally

```bash
git clone https://github.com/Mishi93/nlp_dashboard.git
cd nlp_dashboard
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Requirements: Python 3.10+. No API keys, database, environment file or NLTK download are needed. `data/sample_reviews.csv` and `models/sentiment_model.pkl` must be present in the repo — they already are.

## The data

The dashboard reads one fixed file: `data/sample_reviews.csv` (320 rows).

| Column | What it holds |
| --- | --- |
| `review_id` | REV_0001 – REV_0320 |
| `review_text` | The review text |
| `rating` | 1–5 |
| `category` | Books, Clothing, Electronics, Home & Kitchen |
| `review_date` | January – June 2026 |

The app adds two columns itself: `sentiment_label` and `sentiment_score`.

**There is no upload button in the app**, despite the repo description saying users upload a CSV. To analyse different reviews, replace `data/sample_reviews.csv` with your own file in the same format (same five columns) before running the app.

## Filters

In the sidebar, under **Global Query Filter**:

- **Filter by Domain Group:** All Categories, or one category (Books, Clothing, Electronics, Home & Kitchen)
- **Select Review Window Timeline:** a date range, defaulting to the earliest to latest date in the data

All KPIs, charts, keywords, topics and the table update to the filtered reviews.

## KPIs

Computed on the filtered data:

| KPI | How it is calculated |
| --- | --- |
| Total Volumes Evaluated | Number of reviews in the filtered set |
| Positive Sentiment Ratio | % of reviews labelled Positive |
| Aggregate Satisfaction Score | Mean rating, shown as `x.xx / 5.0` |
| Net Sentiment Margin | (Positive − Negative) / total × 100 |

## Dashboard

Three tabs, one page:

### 1. High-Level Executive Analytics

- **Overall Sentiment Composition** — donut chart of Positive / Neutral / Negative counts (green / grey / red).
- **Sentiment Distribution across Categories** — stacked bar of review counts by category and sentiment.
- **Sentiment Trend (Average Rating History)** — daily average rating over time (y-axis fixed 1–5). Despite the name, this plots average *rating*, not sentiment counts.

### 2. Deep-Dive Text Discovery

- **Top Keywords by Significance (TF-IDF Weighting)** — horizontal bar of the top 12 keywords in the filtered reviews, ranked by summed TF-IDF weight.
- **Unsupervised Latent Topic Clusters (LDA)** — 4 topics, 5 words each, each shown in its own expander. Needs at least 4 reviews in the filtered set.

### 3. Raw Processing Data Ledger

The filtered reviews as a table: `review_id`, `review_date`, `category`, `review_text`, `rating`, `sentiment_label`, `sentiment_score`. There is no search, export or download button.

## How the NLP works

All in `src/`:

- **Preprocessing** (`src/preprocess.py`) — lowercases text, strips HTML tags, removes punctuation and digits, tokenises with a regex and drops a hardcoded English stopword list. No NLTK or spaCy is used.
- **Sentiment** (`src/sentiment.py`) — `SentimentAnalyzer` tries to load `models/sentiment_model.pkl`. If it loads, it calls `predict` / `predict_proba` and maps `{0: Negative, 1: Neutral, 2: Positive}`, with the score as the highest class probability. If the model is missing or fails, it falls back to a hardcoded lexicon: 12 positive words and 13 negative words, with a synthetic score formula — that fallback score is **not** a probability.
- **Keywords** (`src/keyword_extraction.py`) — TF-IDF (`max_df=0.85`, `min_df=1`), top keywords by summed TF-IDF weight across the filtered corpus.
- **Topics** (`src/topic_model.py`) — LDA with `CountVectorizer(max_df=0.95, min_df=2)`, 4 topics, `random_state=42`.
- **Visualisation** (`src/visualization.py`) — all four charts are Plotly. Matplotlib and seaborn are listed in `requirements.txt` but are not used anywhere in the code.

## The model

Trained in `notebooks/model_training.ipynb`:

- **Pipeline:** TF-IDF (`ngram_range=(1, 2)`, `max_features=5000`, `stop_words='english'`) + Logistic Regression (`C=1.0`, `max_iter=1000`, `random_state=42`)
- **Training data:** the same 320-row `sample_reviews.csv`
- **Labels — read this before quoting the accuracy:** labels are derived from the rating, not from human annotation — ratings 1–2 → Negative, 3 → Neutral, 4–5 → Positive.
- **Split:** 80% train / 20% test, stratified, `random_state=42` (64 test rows)
- **Test accuracy in the notebook: 1.00** — precision, recall and F1 of 1.00 for every class

That 1.00 measures how well the model predicts the *rating band* from the review text on this small sample. It is not accuracy against independently labelled sentiment, and it should not be presented as real-world accuracy. Treat it as a sign that the sample is small and the labels come from the ratings, not as proof the model is perfect.

## Screenshots

<img width="1366" height="640" alt="image" src="https://github.com/user-attachments/assets/84bf5a21-fe1f-45d2-9dd1-ee7d10073b06" />

<img width="1366" height="648" alt="image" src="https://github.com/user-attachments/assets/2ba8f9c2-c5f9-4ef7-9790-4baa72594e7e" />

<img width="1366" height="641" alt="image" src="https://github.com/user-attachments/assets/0af70cf2-01e6-4de6-8a87-59b431756d44" />




## Project structure

```
nlp_dashboard/
├── app.py                        # Streamlit dashboard (loads data/sample_reviews.csv)
├── requirements.txt              # Pinned dependencies
├── data/
│   └── sample_reviews.csv        # 320-row sample (the only data source)
├── models/
│   └── sentiment_model.pkl       # Saved sentiment model
├── notebooks/
│   └── model_training.ipynb      # TF-IDF + Logistic Regression training
├── src/
│   ├── preprocess.py             # Cleaning, tokenising, stopwords
│   ├── sentiment.py              # Model sentiment + lexicon fallback
│   ├── keyword_extraction.py     # TF-IDF top keywords
│   ├── topic_model.py            # LDA topics
│   ├── visualization.py          # All Plotly charts
│   └── utils.py                  # load_data, save/load model, ensure_directories
└── README.md
```

## Technologies

- Python 3.10+
- Streamlit 1.35.0
- scikit-learn 1.5.0 (TF-IDF, Logistic Regression, LDA)
- pandas 2.2.2, NumPy 1.26.4
- Plotly 5.22.0

`requirements.txt` also pins matplotlib, seaborn, python-dotenv and requests, but none of those are imported anywhere in the app or `src/`.

## Limitations

Honest notes, so nobody mistakes this for a production system:

- **No upload.** The app always reads `data/sample_reviews.csv`. A user's own CSV can only be analysed by replacing that file before running.
- **No live demo, no deployment.** Nothing is hosted, and there is no Dockerfile or deployment config.
- **The 1.00 accuracy is not what it looks like.** Labels come from the 1–5 rating, the sample is only 320 rows, and some review texts repeat across categories. Do not quote 1.00 as the model's accuracy on real sentiment.
- **The bundled model file is opaque.** The notebook trains TF-IDF + Logistic Regression, but the `.pkl` in `models/` was not verified to be that exact pipeline — the code only says the expected pipeline is "e.g. TF-IDF + LogisticRegression" in a comment.
- **The training notebook will not re-save into the repo as written.** It saves the model to a hardcoded Windows path (`D:\NLP_DASHBOARD\models\sentiment_model.pkl`), not a relative repo path.
- **No tests, licence or `.gitignore`.** There are no tests, and `src/__pycache__/` (8 `.pyc` files) is committed to the repo.
- **The trend chart plots rating, not sentiment.** Its title says "Sentiment Trend" but the line is daily average rating.

## Ideas for next steps

- Add a file uploader (`st.file_uploader`) so a real CSV can be analysed without replacing the sample file — that is the single biggest gap versus the repo description.
- Fix the notebook's save path to a relative `models/sentiment_model.pkl`, and re-train and re-save the model from it so the `.pkl` provably matches the notebook.
- Get sentiment labels from human annotation (or a labelled dataset) instead of rating bands, and report accuracy on that.
- Add a `.gitignore` (drop the committed `__pycache__`), a licence, and a small test for the sentiment fallback and the KPI calculations.
- Rename the trend chart to "Average Rating Over Time", or plot sentiment counts alongside it.
- Deploy the app (for example on Streamlit Community Cloud) and add the live link here.
