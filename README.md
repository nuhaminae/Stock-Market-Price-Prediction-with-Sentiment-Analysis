# Stock Market Price Prediction with Sentiment Analysis

This repository contains Jupyter Notebooks and resources for analysing and predicting stock market prices using both historical price data and sentiment analysis from publication sources.

## Overview

The goal of this project is to explore how sentiment data can enhance traditional time-series forecasting models for stock prices. The analysis covers exploratory data analysis (EDA) of historical prices, sentiment extraction from news sources, and predictive modelling that combines both data streams.

## Features

- **Exploratory Data Analysis (EDA):**  
  Visualizations and statistical summaries of historical stock price movements.

- **Sentiment Analysis:**  
  Extraction of sentiment scores from news headlines or articles related to the target stocks.


## Repository Structure

```
notebooks/
    ├── historical_price_eda.ipynb   # EDA on historical price data
    ├── news_eda.ipynb               # EDA on news/sentiment data
    ├── modeling.ipynb               # Predictive modeling notebooks
data/
    ├── raw/                         # Raw data files (not included)
    ├── processed/                   # Cleaned and processed datasets
requirements.txt                     # Python dependencies
README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/nuhaminae/Stock-Market-Price-Prediction-with-Sentiment-Analysis.git
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3. Open and run the notebooks in the `notebooks/` directory to explore EDA and modeling steps.

## Recent Activity

This project is actively being developed.

See the [commit history](https://github.com/nuhaminae/Stock-Market-Price-Prediction-with-Sentiment-Analysis/commits?per_page=5&sort=updated) for more details.
