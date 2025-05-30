from textblob import TextBlob

def calculate_sentiment(text):
    """
    Calculates the sentiment polarity of a given text.

    Args:
        text (str): The input text.

    Returns:
        float: The sentiment polarity score.
    """
    return TextBlob(text).sentiment.polarity

def add_sentiment_column(df, text_column='headline'):
    """
    Adds a 'sentiment' column to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_column (str): The name of the text column to analyze.

    Returns:
        pandas.DataFrame: The DataFrame with the added 'sentiment' column.
    """
    df['sentiment'] = df[text_column].apply(calculate_sentiment)
    return df