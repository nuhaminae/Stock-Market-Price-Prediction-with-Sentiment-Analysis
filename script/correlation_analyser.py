import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_stock_data(hist_data, senti_data):
    """
    Loads historical price and sentiment data from specified DataFrames.

    Args:
        hist_data (pd.DataFrame): The historical price data DataFrame.
        senti_data (pd.DataFrame): The sentiment data DataFrame.

    Returns:
        tuple: A tuple containing the historical price DataFrame and sentiment DataFrame,
               or (None, None) if input DataFrames are None.
    """
    if hist_data is None or senti_data is None:
        print("Input DataFrames are None. Skipping.")
        return None, None
    return hist_data, senti_data
    
def save_dataframe(ticker, df, df_folder, df_name) -> None:
    """
        Saves a DataFrame to a specified directory as a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            df_folder (str): The directory path where the DataFrame will be saved.
            df_name (str): The name of the CSV file (without extension).
    """
    if not os.path.exists(df_folder):
        os.makedirs(df_folder)

    df_path = os.path.join(df_folder, f'{df_name}.csv')

    #reset the index to include the Date as a column
    df_to_save = df.reset_index()

    #save the DataFrame to the specified directory
    df_to_save.to_csv(df_path, index=False)

    #calculate the relative path
    current_directory = os.getcwd()
    relative_df_path = os.path.relpath(df_path, current_directory)

    print(f'DataFrame saved to: {relative_df_path}\n')

def process_and_align_data(hist_data, senti_data, start_date=None, end_date=None):
    """
    Processes and aligns historical price data with aggregated sentiment data.

    Args:
        hist_data (pd.DataFrame): The historical price data.
        senti_data (pd.DataFrame): The sentiment data.
        start_date (str, optional): The start date for filtering historical data (YYYY-MM-DD).
                                    Defaults to None.
        end_date (str, optional): The end date for filtering historical data (YYYY-MM-DD).
                                  Defaults to None.

    Returns:
        pd.DataFrame: The aligned DataFrame, or None if input data is None.
    """
    if hist_data is None or senti_data is None:
        return None

    senti_data.dropna(inplace=True)
    agg_senti = senti_data.groupby('Date')['Sentiment'].mean().reset_index()
    '''
    if start_date and end_date:
        hist_data = hist_data[
            (hist_data['Date'] >= start_date) &
            (hist_data['Date'] <= end_date)]
        hist_data = hist_data.reset_index(drop=True)
    '''
    hist_data.dropna(inplace=True)

    agg_senti['Date'] = pd.to_datetime(agg_senti['Date']).dt.normalize().dt.tz_localize(None)
    hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.normalize()

    aligned_data = pd.merge(hist_data, agg_senti, left_on='Date', right_on='Date', how='inner')
    aligned_data = aligned_data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore') # Use errors='ignore'

    aligned_data['Sentiment_Category'] = pd.cut(aligned_data['Sentiment'],
                                               bins=[-1.1, -0.00001, 0.00001, 1.1],
                                               labels=['Negative', 'Neutral', 'Positive'],
                                               include_lowest=True)

    # Calculate Daily Return
    aligned_data['Daily_Return'] = aligned_data['Adj Close'].pct_change()
    return aligned_data

def save_plot(plot_folder, plot_name, plot_path):
    """
    Saves the current matplotlib plot to a specified location.

    Args:
        plot_folder (str): The folder to save the plot.
        plot_name (str): The name of the plot file.
        plot_path (str): The full path to save the plot.
    """
    
    #create the directory if it doesn't exist
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    #save plot
    plt.savefig(plot_path)

    #calculate the relative path
    current_directory = os.getcwd()
    relative_plot_path = os.path.relpath(plot_path, current_directory)

    #display message and close plot
    print(f'\nPlot is saved to {relative_plot_path}.\n')
    plt.close()

def plot_sentiment_distribution(aligned_data, ticker,plot_folder):
    """
    Plots the distribution of sentiment categories.

    Args:
        aligned_data (pd.DataFrame): The aligned data DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    if aligned_data is None:
        return

    sentiment_percentages = aligned_data['Sentiment_Category'].value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages.values,
                hue=sentiment_percentages.index, palette='viridis', legend=False)
    plt.title(f'{ticker} - Distribution of Publisher Sentiment Categories')
    plt.xlabel('Sentiment Category')
    plt.ylabel("Percentage of Publishers' Sentiment")
    plt.grid()
    plt.show()

    plot_name = f'{ticker} - Distribution of Publisher Sentiment Categories.png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)

def plot_daily_return_and_sentiment(aligned_data, ticker,plot_folder):
    """
    Plots daily stock returns and sentiment over time.

    Args:
        aligned_data (pd.DataFrame): The aligned data DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    if aligned_data is None:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Daily Returns
    ax1.plot(aligned_data['Date'], aligned_data['Daily_Return'], color='black', label='Daily Return')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Return', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for Sentiment
    ax2 = ax1.twinx()
    ax2.plot(aligned_data['Date'], aligned_data['Sentiment'], color='red', label='Sentiment')
    ax2.set_ylabel('Sentiment', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'{ticker} - Daily Stock Returns and Sentiment Over Time')
    fig.tight_layout()
    plt.show()

    plot_name = f'{ticker} - Daily Stock Returns and Sentiment Over Time.png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)

def plot_sentiment_vs_daily_return(aligned_data, ticker,plot_folder):
    """
    Plots a scatter plot of sentiment vs. daily return.

    Args:
        aligned_data (pd.DataFrame): The aligned data DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    if aligned_data is None:
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=aligned_data, x='Sentiment', y='Daily_Return')
    plt.title(f'{ticker} - Scatter Plot of Sentiment vs. Daily Return')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return')
    plt.show()

    plot_name = f'{ticker} - Scatter Plot of Sentiment vs. Daily Return.png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)
    
def plot_sentiment_vs_volatility(aligned_data, ticker,plot_folder):
    """
    Plots a scatter plot of sentiment vs. volatility.

    Args:
        aligned_data (pd.DataFrame): The aligned data DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    if aligned_data is None or 'Volatility' not in aligned_data.columns:
        if aligned_data is not None:
             print(f"Volatility column not found or data is None for {ticker}. Skipping Volatility analysis.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=aligned_data, x='Sentiment', y='Volatility')
    plt.title(f'{ticker} - Scatter Plot of Sentiment vs. Volatility (Rolling Std Dev)')
    plt.xlabel('Aggregated Sentiment Score')
    plt.ylabel('Volatility (Rolling Std Dev)')
    plt.show()

    plot_name = f'{ticker} - Scatter Plot of Sentiment vs. Volatility (Rolling Std Dev).png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)

def analyse_and_plot(ticker, hist_data, senti_data, start_date=None, end_date=None, plot_folder=None):
    """
    Performs the full analysis and plotting for a given ticker
    using specified data DataFrames.

    Args:
        ticker (str): The stock ticker symbol.
        hist_data (pd.DataFrame): The historical price data DataFrame.
        senti_data (pd.DataFrame): The sentiment data DataFrame.
        start_date (str, optional): The start date for filtering historical data (YYYY-MM-DD).
                                    Defaults to None.
        end_date (str, optional): The end date for filtering historical data (YYYY-MM-DD).
                                  Defaults to None.
        plot_folder (str): The folder to save the plot.
    """
    print(f"Analysing data for {ticker}...")
    hist_data, senti_data = load_stock_data(hist_data, senti_data)

    aligned_data = process_and_align_data(hist_data, senti_data, start_date, end_date)

    if aligned_data is not None:
        plot_sentiment_distribution(aligned_data, ticker, plot_folder)
        plot_daily_return_and_sentiment(aligned_data, ticker, plot_folder)
        plot_sentiment_vs_daily_return(aligned_data, ticker, plot_folder)
        plot_sentiment_vs_volatility(aligned_data, ticker, plot_folder)

        # Correlation Analysis
        correlation = aligned_data['Sentiment'].corr(aligned_data['Daily_Return'])
        print(f"{ticker} - Correlation between news sentiment and daily stock returns: {correlation}")

        correlation_matrix = aligned_data[['Sentiment', 'Daily_Return', 'Adj Close']].corr()

        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
        plt.title(f'{ticker} - Correlation Heatmap of Sentiment and Daily Return')
        plt.show()

        plot_name = f'{ticker} - Correlation Heatmap of Sentiment and Daily Return.png'
        plot_path = os.path.join(plot_folder, plot_name)

        #save plot
        save_plot(plot_folder, plot_name, plot_path)