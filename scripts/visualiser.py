import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def save_plot(plot_folder, plot_name, plot_path):
    """
    Saves the current matplotlib plot to a specified location.

    Args:
        plot_folder (str): The folder to save the plot.
        plot_name (str): The name of the plot file.
        plot_path (str): The full path to save the plot.
    """
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(plot_path)
    print(f'\nPlot is saved to {plot_path}.\n')

def plot_publication_frequency_by_day(df, ticker, plot_folder):
    """
    Plots and saves the publication frequency by day of the week.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    df['publication_day'] = df['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    publication_day_counts = df['publication_day'].value_counts().reindex(day_order)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=publication_day_counts.index, y=publication_day_counts.values,
                hue=publication_day_counts.index, palette='viridis', legend=False)
    plt.title(f'{ticker} - Number of Publications by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_name = f'{ticker} - Number of Publications by Day of the Week.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    plt.show()
    plt.close()

def plot_sentiment_distribution(df, ticker, plot_folder):
    """
    Plots and saves the distribution of sentiment scores.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(df['sentiment'], bins=20, kde=True)
    plt.title(f'{ticker} - Distribution of Sentiment Scores Headlines')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.gca().spines[['top', 'right']].set_visible(False)

    plot_name = f'{ticker} - Distribution of Sentiment Scores Headlines.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    plt.show()
    plt.close()

def plot_daily_publication_frequency(df, ticker, plot_folder):
    """
    Plots and saves the daily article publication frequency.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    daily_publications = df.resample('D', on='date').size()

    plt.figure(figsize=(12, 6))
    daily_publications.plot()
    plt.title(f'{ticker} - Daily Article Publication Frequency')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)

    plot_name = f'{ticker} - Daily Article Publication Frequency.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    plt.show()
    plt.close()

def plot_hourly_publication_frequency_zero_hour(df, ticker, plot_folder):
    """
    Plots and saves the hourly publication frequency for the 00:00 hour.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    df['publish_hour'] = pd.to_datetime(df['date']).dt.hour
    zero_hour_data = df[df['publish_hour'] == 0]
    zero_hour_count = len(zero_hour_data)
    print(f'\nNumber of articles published at 00:00 is {zero_hour_count}\n')

    plot_data = pd.DataFrame({'hour': ['00:00'], 'count': [zero_hour_count]})

    plt.figure(figsize=(6, 4))
    sns.barplot(x='hour', y='count', data=plot_data, hue='hour', palette='viridis', legend=False)
    plt.title(f'{ticker} - Article Publishing Frequency at 00:00 Hour')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Articles')
    plt.grid(True)

    plot_name = f'{ticker} - Article Publishing Frequency at 00 00 Hour.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    plt.show()
    plt.close()

def plot_hourly_publication_frequency_other_hours(df, ticker, plot_folder):
    """
    Plots and saves the hourly publication frequency for hours other than 00:00.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    df['publish_hour'] = pd.to_datetime(df['date']).dt.hour
    other_hours_data = df[df['publish_hour'] != 0]
    hourly_counts = other_hours_data['publish_hour'].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values,
                hue=hourly_counts.index, palette='viridis', legend=False)
    plt.title(f'{ticker} - Article Publishing Frequency Hours (Excluding 00:00 Hour)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Articles')
    plt.xticks(range(24)) # Ensure all hours are shown if present
    plt.grid(True)

    plot_name = f'{ticker} - Article Publishing Frequency Hours (Excluding 00 00 Hour).png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    plt.show()
    plt.close()