import pandas as pd
from . import sentiment_analyser
from . import text_processor
from . import visualiser
import os

def analyze_stock_news(df, ticker, plot_folder):
    """
    Performs a comprehensive analysis of news headlines for a given stock ticker.

    Args:
        df (pandas.DataFrame): The DataFrame containing news for the ticker.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plots.
    """
    print(f"\n--- Analyzing {ticker} News Headlines ---")

    # Descriptive Statistics
    print("\nDescriptive statistics for 'headline' column:")
    print(df['headline'].describe())
    print("\nNumber of unique headlines:")
    print(df['headline'].nunique())
    df['headline_length'] = df['headline'].apply(len)
    print("\nBasic statistics for 'headline_length' column:")
    print(df['headline_length'].describe())
    print("\nMost frequent headlines:")
    print(df['headline'].value_counts().head())
    print('\nNumber of articles per publisher:\n')
    print(df['publisher'].value_counts())

    # Publication Analysis
    print("\nPublication trends by day of the week:")
    print(df['date'].dt.day_name().value_counts())
    visualiser.plot_publication_frequency_by_day(df, ticker, plot_folder)

    # Sentiment Analysis
    df = sentiment_analyser.add_sentiment_column(df)
    print('\nSentiment distribution:')
    print(df['sentiment'].describe())
    print("\nMost Positive Headlines:")
    print(df.nlargest(5, 'sentiment')[['headline', 'sentiment']])
    print("\nMost Negative Headlines:")
    print(df.nsmallest(5, 'sentiment')[['headline', 'sentiment']])
    visualiser.plot_sentiment_distribution(df, ticker, plot_folder)

    # Text Analysis (Topic Modelling)
    text_data = df['headline'].dropna().tolist()
    processed_text_data = [text_processor.preprocess_text(text) for text in text_data]

    print('\nTop TF-IDF terms:\n')
    tfidf_matrix, feature_names = text_processor.calculate_tfidf(processed_text_data)
    average_tfidf = tfidf_matrix.mean(axis=0).A1
    sorted_tfidf_indices = average_tfidf.argsort()[::-1]
    top_tfidf_terms = [(feature_names[i], average_tfidf[i]) for i in sorted_tfidf_indices[:20]]
    for term, score in top_tfidf_terms:
        print(f'{term}: {score:.4f}')

    text_processor.perform_lda_topic_modeling(tfidf_matrix, feature_names)
    text_processor.perform_ner(text_data)

    # Time Series Analysis
    visualiser.plot_daily_publication_frequency(df, ticker, plot_folder)
    visualiser.plot_hourly_publication_frequency_zero_hour(df, ticker, plot_folder)
    visualiser.plot_hourly_publication_frequency_other_hours(df, ticker, plot_folder)

    # Publisher Analysis
    sentiment_by_publisher = df.groupby('publisher')['sentiment'].mean()
    print('\nAverage sentiment by publisher:\n')
    print(sentiment_by_publisher)

    df['domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'Not an email')
    domain_counts = df['domain'].value_counts()
    print('\nDomains with the most contributions:\n')
    print(domain_counts[domain_counts.index != 'Not an email'].head())

    print(f"\n--- Analysis for {ticker} Complete ---")