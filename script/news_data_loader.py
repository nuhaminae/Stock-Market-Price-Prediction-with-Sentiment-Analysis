#important python libraries
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
        
        #change 'FB' ticker name to 'META' and 'MSF' to 'MSFT' for uniformity
        text_data.loc[text_data['stock'] == 'FB', 'stock'] = 'META'
        text_data.loc[text_data['stock'] == 'MSF', 'stock'] = 'MSFT'

        print("File loaded successfully. The new DataFrame is 'stock_news'.")

        #filter text_data dataframe to include rows where the 'stock' column is in the 'tickers' list
        #drop 'Unnamed: 0'column, reset and drop old index, and sort index
        stock_news = text_data[text_data['stock'].isin(tickers)].drop('Unnamed: 0',
                                                                      axis='columns').reset_index(drop=True).sort_index(axis=1)
        
        #pass `format='ISO8601'` to change some date formats that are `'format='%Y-%m-%d %0:%0:%0%0'`
        stock_news['date'] = pd.to_datetime(stock_news['date'],
                                            format='ISO8601', errors='coerce')
        
        #reverse, reset, and drop index 
        stock_news = stock_news.reindex(stock_news.index[::-1])
        stock_news = stock_news.reset_index(drop=True)

        #capitalise the first letter of the column names

        stock_news.columns = [col.capitalize() for col in stock_news.columns]
        return stock_news

    except FileNotFoundError:
        print(f"Error: The file path '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f'An error occurred while loading file: {e}')
        return None