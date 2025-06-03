import pandas as pd
import os

from typing import Optional, Union, List, Dict

# for visualisation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import TA-Lib Library
import ta
from ta.volatility import average_true_range



class StockAnalyser:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialise the StockAnalyser with a pandas DataFrame containing stock data.

        Args:
            dataframe (pd.DataFrame): DataFrame with stock data.
                                      It should contain 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
                                      'Date' column should be in datetime format.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not all(col in dataframe.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
             raise ValueError("DataFrame must contain 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume' columns.")
        if not pd.api.types.is_datetime64_any_dtype(dataframe['Date']):
             raise ValueError("'Date' column must be in datetime format.")

        self.data = dataframe.copy()
        self.data = self.data.set_index('Date') # Set Date as index for easier handling

    def get_historical_data(self) -> pd.DataFrame:
        """
        Return the historical stock data loaded into the analyser.

        Returns:
            pd.DataFrame: Historical stock data
        """
        return self.data

    def calculate_technical_indicators(self, data) -> pd.DataFrame:
        """
        Calculate basic technical indicators for the stock data.

        Args:
            data (pd.DataFrame): Historical stock data

        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        df = data.copy()

        #calculate simple moving average using ta library
        df['SMA_15'] = ta.trend.sma_indicator(df['Adj Close'], window=10) #10 days
        df['SMA_60'] = ta.trend.sma_indicator(df['Adj Close'], window=20) #20 days

        #calculate RSI (Relative Strength Index) using ta library
        df['RSI_15'] = ta.momentum.rsi(df['Adj Close'], window=10) #10 days
        df['RSI_60'] = ta.momentum.rsi(df['Close'], window=20) #20 days

        #calculate MACD (Moving Average Convergence Divergence) using ta library
        df['MACD'] = ta.trend.macd(df['Adj Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Adj Close'])
        df['MACD_Diff'] = ta.trend.macd_diff(df['Adj Close'])

        #calculate volatality (Average True Range/ATR) using ta library
        df['Volatility'] = average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=10)
        
        return df
        
    def plot_stock_data(
                self,
                data: pd.DataFrame, #this is the DataFrame to be used for plotting
                ticker: str,
                indicators: bool = True,
                volume: bool = True,
                volume_color: str = 'black'
            ) -> go.Figure:
                """
                Create an interactive plot of stock data using Plotly.

                Args:
                    data (pd.DataFrame): Historical stock data (should already have indicators if needed).
                    ticker (str): Stock ticker symbol for the title
                    indicators (bool): Whether to show technical indicators (assumes they are in the data DataFrame).
                    volume (bool): Whether to show volume data

                Returns:
                    go.Figure: The Plotly figure object.
                """
                #ensure the Date index is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(data.index):
                     raise ValueError("DataFrame index must be in datetime format for plotting.")


                #create figure with secondary y-axis
                fig = make_subplots(
                    rows=2 if volume else 1,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3] if volume else [1]
                )

                #add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )

                if indicators:
                    #check if indicator columns exist in the data
                    required_indicators = ['SMA_15', 'SMA_60']
                    if not all(ind in data.columns for ind in required_indicators):
                        print(f"Warning: Indicator columns {required_indicators} not found in the provided data. Skipping indicator plot.")
                    else:
                        #add moving averages
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['SMA_15'],
                                name='SMA 15',
                                line=dict(color='blue')
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['SMA_60'],
                                name='SMA 60',
                                line=dict(color='red')
                            ),
                            row=1, col=1
                        )

                if volume:
                    #check if Volume column exists
                    if 'Volume' not in data.columns:
                         print("Warning: 'Volume' column not found in the provided data. Skipping volume plot.")
                    else:
                        #add volume bar chart
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['Volume'],
                                name='Volume',
                                marker_color=volume_color
                            ),
                            row=2, col=1
                        )

                #update layout
                fig.update_layout(
                    title=f'{ticker} - Historical Stock Price',
                    yaxis_title='Price',
                    xaxis_title='Date',
                    template='plotly_white'
                )

                return fig #return the figure object

    def save_plot(self, fig, plot_folder, plot_name) -> None:
        """
        Saves the Plotly figure to a local directory.

        Args:
            fig (go.Figure): The Plotly figure object to save.
            plot_folder (str): The folder path where the plot will be saved.
            plot_name (str): The name of the plot file (without extension).
        """
        if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

        plot_path = os.path.join(plot_folder, f'{plot_name}.html')

        #save plot as HTML
        fig.write_html(plot_path)
         
        #calculate the relative path
        current_directory = os.getcwd()
        relative_plot_path = os.path.relpath(plot_path, current_directory)

        print(f'Plot saved to: {relative_plot_path}')

    def save_dataframe(self, df, df_folder, df_name) -> None:
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

        # Reset the index to include the Date as a column
        df_to_save = df.reset_index()

        #save the DataFrame to the specified directory
        df_to_save.to_csv(df_path, index=False)

        #calculate the relative path
        current_directory = os.getcwd()
        relative_df_path = os.path.relpath(df_path, current_directory)

        print(f'DataFrame saved to: {relative_df_path}\n')