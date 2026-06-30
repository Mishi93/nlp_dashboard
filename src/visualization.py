# nlp-dashboard/src/visualization.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class DashboardVisualizer:
    def __init__(self):
        # Professional dark/light adaptive template configuration
        self.theme_colors = {
            'Positive': '#2ecc71',
            'Neutral': '#95a5a6',
            'Negative': '#e74c3c'
        }

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Generates a clean pie/donut chart representing categorical distribution."""
        counts = df['sentiment_label'].value_counts().reset_index()
        counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(
            counts, 
            values='Count', 
            names='Sentiment',
            hole=0.4,
            color='Sentiment',
            color_discrete_map=self.theme_colors,
            title='Overall Sentiment Composition'
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
        return fig

    def plot_sentiment_over_time(self, df: pd.DataFrame) -> go.Figure:
        """Plots rolling aggregations of trends over chronological timeline indices."""
        # Ensure sorting matches sequence arrays
        df_sorted = df.sort_values('review_date')
        
        # Daily breakdown tracking of mean performance rating metrics
        timeline = df_sorted.groupby('review_date').agg(
            avg_rating=('rating', 'mean'),
            volume=('review_id', 'count')
        ).reset_index()
        
        fig = px.line(
            timeline, 
            x='review_date', 
            y='avg_rating',
            title='Sentiment Trend (Average Rating History)',
            labels={'review_date': 'Timeline Date', 'avg_rating': 'Avg Rating'},
            markers=True
        )
        fig.update_traces(line_color='#3498db', line_width=2)
        fig.update_layout(yaxis_range=[1, 5], margin=dict(t=40, b=20, l=20, r=20))
        return fig

    def plot_top_keywords(self, keywords: list) -> go.Figure:
        """Visualizes extracted terms and relative frequency metric values."""
        if not keywords:
            # Empty structural placeholder figure array
            return go.Figure()
            
        df_kw = pd.DataFrame(keywords, columns=['Keyword', 'Importance'])
        df_kw = df_kw.sort_values(by='Importance', ascending=True)
        
        fig = px.bar(
            df_kw, 
            x='Importance', 
            y='Keyword', 
            orientation='h',
            title='Top Keywords by Significance Metric (TF-IDF Weighting)',
            labels={'Importance': 'Relative Cumulative Mass'},
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
        return fig

    def plot_category_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """Tracks normalized sub-distribution vectors grouped across market categories."""
        cat_sentiment = df.groupby(['category', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        
        # Guard column variants if a category distribution misses explicit instances
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment not in cat_sentiment.columns:
                cat_sentiment[sentiment] = 0
                
        fig = go.Figure()
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=cat_sentiment['category'],
                y=cat_sentiment[sentiment],
                marker_color=self.theme_colors[sentiment]
            ))
            
        fig.update_layout(
            barmode='stack',
            title='Sentiment Distribution across Categories',
            xaxis_title='Product Category',
            yaxis_title='Review Count',
            margin=dict(t=40, b=20, l=20, r=20)
        )
        return fig