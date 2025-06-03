#important python libraries
import pandas as pd
import os

#for visualisation
import matplotlib.pyplot as plt
import seaborn as sns


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

    #display message
    print(f'\nPlot is saved to {relative_plot_path}.\n')

def plot_publication_frequency_by_day(df, ticker, plot_folder):
    """
    Plots and saves the publication frequency by day of the week.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    #analyse publication dates
    df['Publication_Day'] = df['Date'].dt.day_name()
    print("\nPublication trends by day of the week:")
    print(df['Publication_Day'].value_counts())

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    publication_day_counts = df['Publication_Day'].value_counts().reindex(day_order)

    #plot publication dates and save plot image 
    plt.figure(figsize=(10, 6))
    sns.barplot(x=publication_day_counts.index, y=publication_day_counts.values,
                hue=publication_day_counts.index, palette='viridis', legend=False)
    plt.title(f'{ticker} - Number of Publications by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.tight_layout()

    #select plot directory and plot name to save plot
    plot_name = f'{ticker} - Number of Publications by Day of the Week.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()

    #close plot to free up memory
    plt.close()

def plot_sentiment_distribution(df, ticker, plot_folder):
    """
    Plots and saves the distribution of sentiment scores.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    #plot the distribution of sentiment scores
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Sentiment'], bins=20, kde=True)
    plt.title(f'{ticker} - Distribution of Sentiment Scores Headlines')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.gca().spines[['top', 'right']].set_visible(False)

    plot_name = f'{ticker} - Distribution of Sentiment Scores Headlines.png'
    plot_path = os.path.join(plot_folder, plot_name)
    save_plot(plot_folder, plot_name, plot_path)
    
    #show plot
    plt.show()
    
    #close plot to free up memory
    plt.close()

def plot_daily_publication_frequency(df, ticker, plot_folder):
    """
    Plots and saves the daily article publication frequency.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    ##plot and save the daily publication frequency
    daily_publications = df.resample('D', on='Date').size()

    #plot
    plt.figure(figsize=(12, 6))
    daily_publications.plot()
    plt.title(f'{ticker} - Daily Article Publication Frequency')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid()

    plot_name = f'{ticker} - Daily Article Publication Frequency.png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()

    #close plot to free up space
    plt.close()

def plot_hourly_publication_frequency_zero_hour(df, ticker, plot_folder):
    """
    Plots and saves the hourly publication frequency for the 00:00 hour.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    #extract the hour from the datetime column
    df['Publication_Hour'] = pd.to_datetime(df['Date']).dt.hour

    #filter data for 00:00
    zero_hour_data = df[df['Publication_Hour'] == 0]

    #calculate and print the count of articles published at 00:00
    zero_hour_count = len(zero_hour_data)
    print(f'\nNumber of articles published at 00:00 is {zero_hour_count}\n')

    #plot for 00:00
    #Create a small DataFrame 
    plot_data = pd.DataFrame({'hour': ['00:00'], 'count': [zero_hour_count]})

    plt.figure(figsize=(6, 4))
    sns.barplot(x='hour', y='count', data=plot_data, hue='hour', palette='viridis', legend=False)
    plt.title(f'{ticker} - Article Publishing Frequency at 00:00 Hour')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Articles')
    plt.grid()

    plot_name = f'{ticker} - Article Publishing Frequency at 00 00 Hour.png'
    plot_path = os.path.join(plot_folder, plot_name)
    
    #save plot
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()

    #close plot to free up space
    plt.close()

def plot_hourly_publication_frequency_other_hours(df, ticker, plot_folder):
    """
    Plots and saves the hourly publication frequency for hours other than 00:00.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ticker (str): The stock ticker symbol.
        plot_folder (str): The folder to save the plot.
    """
    #extract the hour from the datetime column
    df['Publication_Hour'] = pd.to_datetime(df['Date']).dt.hour

    #plot for 00:00
    #Count articles per hour for other times
    other_hours_data = df[df['Publication_Hour'] != 0]
    hourly_counts = other_hours_data['Publication_Hour'].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values,
                hue=hourly_counts.index, palette='viridis', legend=False)
    plt.title(f'{ticker} - Article Publishing Frequency Hours (Excluding 00:00 Hour)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Articles')
    plt.grid()

    plot_name = f'{ticker} - Article Publishing Frequency Hours (Excluding 00 00 Hour).png'
    plot_path = os.path.join(plot_folder, plot_name)

    #save plot
    save_plot(plot_folder, plot_name, plot_path)

    #show plot
    plt.show()

    #close plot to free up space
    plt.close()