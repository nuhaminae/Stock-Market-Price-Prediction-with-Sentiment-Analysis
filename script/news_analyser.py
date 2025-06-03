from . import news_sentiment_analyser
from . import news_text_processor
from . import news_visualiser

def analyse_stock_news(df, ticker, plot_folder):
    """
    Performs a comprehensive analysis of news headlines for a given stock ticker.

    Args:
        df (pandas.DataFrame): The DataFrame containing news for the ticker.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plots.
    """
    print(f'\n--- Analysing {ticker} News Headlines ---\n')

    #----Descriptive Statistics-----#
    #display descriptive statistics for the 'Headline' column
    print("\nDescriptive statistics for 'Headline' column:")
    print(df['Headline'].describe())
    print()

    #get the number of unique headlines
    print('\nNumber of unique headlines:')
    print(df['Headline'].nunique())
    print()

    #display the most frequent headlines
    print('\nMost frequent headlines (head):')
    print(df['Headline'].value_counts().head())
    print()

    #calculate headline length
    df['Headline_Length'] = df['Headline'].apply(len)
    print("\nBasic statistics for 'Headline_Length' column:")
    print(df['Headline_Length'].describe())
    print()

    #count the number of articles per publisher 
    print('\nNumber of articles per publisher (head):')
    print(df['Publisher'].value_counts().head())
    print()
    #------------------------------------------#

    #----Publication Analysis-----#

    #call plot_publication_frequency_by_day function
    #calculate publicatoin trend by the day of the week and save to 'publication_day' column
    #plot and save number of daily publications
    news_visualiser.plot_publication_frequency_by_day(df, ticker, plot_folder)
    print()
    #------------------------------------------#

    #----Sentiment Analysis-----#
    #calculate distribution of sentiment scores
    df = news_sentiment_analyser.add_sentiment_column(df)
    print('\nSentiment distribution:')
    print(df['Sentiment'].describe())
    print()

    #analyse the sentiment of the most positive and negative headlines
    #most positive headlines
    print('\nMost Positive Headlines:')
    print(df.nlargest(5, 'Sentiment')[['Headline', 'Sentiment']])
    print()

    #most negative headlines
    print("\nMost Negative Headlines:")
    print(df.nsmallest(5, 'Sentiment')[['Headline', 'Sentiment']])
    news_visualiser.plot_sentiment_distribution(df, ticker, plot_folder)
    print()
    #------------------------------------------#

    #----Text Analysis (Topic Modelling)-----#
    #preproces dataframe
        #converting to lowercase
        #removing non-alphanumeric characters (except spaces)
        #tokenizing the text into words
        #removing common English stop words
        #lemmatizing words to their base form
    text_data = df['Headline'].dropna().tolist()
    processed_text_data = [news_text_processor.preprocess_text(text) for text in text_data]

    #iterate through each articles to find overall important terms
    #terms with highest TF-IDF scores across all documents
    print('\nTop 10 TF-IDF terms:')

    #calculate Average TF-IDF scores
    #'.A1' converts the result matrix of means into a 1-dimensional array
    tfidf_matrix, feature_names = news_text_processor.calculate_tfidf(processed_text_data)
    average_tfidf = tfidf_matrix.mean(axis=0).A1

    #sort and get top terms
    sorted_tfidf_indices = average_tfidf.argsort()[::-1]
    top_tfidf_terms = [(feature_names[i], average_tfidf[i]) for i in sorted_tfidf_indices[:10]] #top 10
    for term, score in top_tfidf_terms:
        print(f'{term}: {score:.4f}')

    news_text_processor.perform_lda_topic_modeling(tfidf_matrix, feature_names)
    print()

    news_text_processor.perform_ner(text_data)
    print()
    #------------------------------------------#

    #----Time Series Analysis----#
    #call plot_daily_publication_frequency function
    #plot and save the daily publication frequency 
    news_visualiser.plot_daily_publication_frequency(df, ticker, plot_folder)
    print()

    #call plot_hourly_publication_frequency_zero_hour function
    #plot and save the hourly publication frequency for 00:00 hour
    news_visualiser.plot_hourly_publication_frequency_zero_hour(df, ticker, plot_folder)
    print()

    #call plot_hourly_publication_frequency_other_hours function
    #plot and save the hourly publication frequency for hour other than 00:00 hour
    news_visualiser.plot_hourly_publication_frequency_other_hours(df, ticker, plot_folder)
    print()
    #------------------------------------------#

    #----Publisher Analysis----#
    #identify unique domains to check if  publisher is email adress instead of name
    #extract the domain from each email address in the 'Publisher' column
    df['Domain'] = df['Publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'Not an email')
    
    #count the occurrences of each unique domain
    domain_counts = df['Domain'].value_counts()

    #display the domains with the highest counts (excluding 'Not an email' if it exists)
    print('Domains with the most contributions:\n')
    print(domain_counts[domain_counts.index != 'Not an email'].head())

    print(f'\n--- Analysis for {ticker} Complete ---\n')
    #------------------------------------------#