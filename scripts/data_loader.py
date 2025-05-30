import pandas as pd
import os

def load_and_filter_data(file_path, tickers):
    """
    Loads the raw data and filters it for specified stock tickers.

    Args:
        file_path (str): The path to the raw data CSV file.
        tickers (list): A list of stock ticker symbols.

    Returns:
        pandas.DataFrame: The filtered and processed DataFrame.
    """
    try:
        text_data = pd.read_csv(file_path, engine='python')
        print('File loaded successfully')

        stock_news = text_data[text_data['stock'].isin(tickers)].drop('Unnamed: 0',
                                                                      axis='columns').reset_index(drop=True).sort_index(axis=1)
        stock_news['date'] = pd.to_datetime(stock_news['date'],
                                            format='ISO8601', errors='coerce')
        return stock_news

    except FileNotFoundError:
        print(f"Error: The file path '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f'An error occurred while loading file: {e}')
        return None