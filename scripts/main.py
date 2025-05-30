import pandas as pd
from data_loader import load_and_filter_data
from scripts.analyser import analyze_stock_news

if __name__ == "__main__":
    # Define file path and tickers
    text_path = 'C:/Users/nuhamin/Documents/kifiya/week 1/' \
                'Stock-Market-Price-Prediction-with-Sentiment-Analysis/' \
                'data/raw_analyst_ratings.csv'
    tickers = ['AAPL', 'AMZN', 'GOOG', 'FB', 'MSF', 'NVDA', 'TSLA']
    plot_folder = 'C:/Users/nuhamin/Documents/kifiya/week 1/' \
                  'Stock-Market-Price-Prediction-with-Sentiment-Analysis/' \
                  'plot images/news plot'

    # Load and filter data
    stock_news = load_and_filter_data(text_path, tickers)

    if stock_news is not None:
        # Create dataframes for each ticker
        stock_dfs = {}
        for ticker in tickers:
            stock_dfs[ticker] = stock_news[stock_news['stock'].isin([ticker])].reset_index(drop=True).copy()
            # It's good practice to work on copies when creating subsets to avoid SettingWithCopyWarning

        # Analyze each stock news
        for ticker in tickers:
            if ticker in stock_dfs:
                analyze_stock_news(stock_dfs[ticker], ticker, plot_folder)
            else:
                print(f"No data found for ticker: {ticker}")
    else:
        print("Failed to load data. Exiting.")