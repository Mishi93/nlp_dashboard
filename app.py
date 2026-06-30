# nlp-dashboard/app.py
import streamlit as st
import pandas as pd
from datetime import datetime

# Import modular pipeline components
from src.utils import ensure_directories, load_data
from src.sentiment import SentimentAnalyzer
from src.topic_model import TopicModeler
from src.keyword_extraction import KeywordExtractor
from src.visualization import DashboardVisualizer

# Page configuration
st.set_page_config(
    page_title="Enterprise NLP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize project directories on spin-up
ensure_directories()

# Initialize core computational classes
@st.cache_resource
def init_processors():
    return (
        SentimentAnalyzer(),
        TopicModeler(n_topics=4, n_words=5),
        KeywordExtractor(max_features=30),
        DashboardVisualizer()
    )

analyzer, modeler, extractor, visualizer = init_processors()

# App Title & Layout Elements
st.title("📊 Enterprise Customer Sentiment & Text Analytics Dashboard")
st.markdown("---")

# Data loading pipeline
@st.cache_data
def get_processed_data():
    raw_df = load_data("data/sample_reviews.csv")
    # Run full vectorized analysis upfront
    processed_df = analyzer.analyze_dataframe(raw_df, 'review_text')
    processed_df['review_date'] = pd.to_datetime(processed_df['review_date']).dt.date
    return processed_df

try:
    data_df = get_processed_data()
except FileNotFoundError:
    st.error("🚨 `sample_reviews.csv` not found in the `data/` directory. Please populate the data directory first.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📊 Global Query Filter")

# Category selection
categories = ["All Categories"] + list(data_df['category'].unique())
selected_category = st.sidebar.selectbox("Filter by Domain Group:", categories)

# Date filter range bounding
min_date = data_df['review_date'].min()
max_date = data_df['review_date'].max()
selected_dates = st.sidebar.date_input(
    "Select Review Window Timeline:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Apply global query mutations to underlying dataframe
filtered_df = data_df.copy()

if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df['category'] == selected_category]

if len(selected_dates) == 2:
    start_d, end_d = selected_dates
    filtered_df = filtered_df[(filtered_df['review_date'] >= start_d) & (filtered_df['review_date'] <= end_d)]

# --- TOP-LEVEL KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Volumes Evaluated", f"{len(filtered_df)}")
with col2:
    pos_pct = (filtered_df['sentiment_label'] == 'Positive').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.metric("Positive Sentiment Ratio", f"{pos_pct:.1f}%")
with col3:
    avg_rating = filtered_df['rating'].mean() if len(filtered_df) > 0 else 0
    st.metric("Aggregate Satisfaction Score", f"{avg_rating:.2f} / 5.0")
with col4:
    net_score = ((filtered_df['sentiment_label'] == 'Positive').sum() - (filtered_df['sentiment_label'] == 'Negative').sum()) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.metric("Net Sentiment Margin", f"{net_score:+.1f}%")

st.markdown("##")

# --- EXECUTIVE DASHBOARD WORKSPACE TABS ---
tab_analytics, tab_deep_dive, tab_data_viewer = st.tabs([
    "📈 High-Level Executive Analytics", 
    "🕵️ Deep-Dive Text Discovery", 
    "📋 Raw Processing Data Ledger"
])

with tab_analytics:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(visualizer.plot_sentiment_distribution(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(visualizer.plot_category_breakdown(filtered_df), use_container_width=True)
        
    st.markdown("---")
    st.plotly_chart(visualizer.plot_sentiment_over_time(filtered_df), use_container_width=True)

with tab_deep_dive:
    st.subheader("🔑 High-Significance Target Keyword Frequency Matrix")
    corpus = filtered_df['review_text'].tolist()
    
    col_kw, col_topic = st.columns([1, 1])
    
    with col_kw:
        if corpus:
            top_kws = extractor.extract_top_keywords(corpus, top_n=12)
            st.plotly_chart(visualizer.plot_top_keywords(top_kws), use_container_width=True)
        else:
            st.info("No text payload available for current criteria selections.")
            
    with col_topic:
        st.subheader("🧩 Unsupervised Latent Topic Clusters (LDA)")
        st.caption("Underlying algorithmic structure identifying hidden thematic trends:")
        
        if len(corpus) >= 4:
            topics = modeler.fit_extract_topics(corpus)
            for topic_title, words in topics.items():
                with st.expander(f"📌 {topic_title} Key Associations", expanded=True):
                    st.write(", ".join([f"**{word}**" for word in words]))
        else:
            st.info("Insufficient data subset volume to properly align semantic topic matrices.")

with tab_data_viewer:
    st.subheader("💾 Complete Tracked Customer Transaction Subsets")
    st.dataframe(
        filtered_df[['review_id', 'review_date', 'category', 'review_text', 'rating', 'sentiment_label', 'sentiment_score']],
        use_container_width=True,
        hide_index=True
    )