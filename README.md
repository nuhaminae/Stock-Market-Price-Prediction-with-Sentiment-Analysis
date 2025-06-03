# Stock Market Price Prediction with Sentiment Analysis

This repository provides a framework for analyzing and predicting stock market prices by combining traditional historical price data with sentiment analysis from news and publication sources. It is designed for experimentation and research in the intersection of natural language processing (NLP) and quantitative financial analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The goal of this project is to investigate whether sentiment data, extracted from news headlines and articles, can improve the accuracy of stock price forecasting models when combined with traditional time-series data. The project includes:

- Exploratory Data Analysis (EDA) on both price and textual sentiment data.
- Data preprocessing and feature engineering.
- Machine learning and deep learning modeling.
- Evaluation and visualization of results.

## Features

- **Exploratory Data Analysis (EDA):**
  - Visualizations and statistical explorations of historical stock price trends.
  - Outlier detection, volatility analysis, and correlation studies.
- **Sentiment Analysis:**
  - Extraction and scoring of sentiment from financial news and social media.
  - Integration of sentiment scores with price data for feature engineering.
- **Predictive Modeling:**
  - Baseline time-series forecasting (ARIMA, LSTM, etc.).
  - Hybrid models that include sentiment as an input.
- **Visualization:**
  - Interactive charts (matplotlib, seaborn, plotly) for both EDA and model results.

## Project Structure

```
notebooks/
    ├── historical_price_eda.ipynb   # EDA on historical price data
    ├── news_eda.ipynb               # EDA on news/sentiment data
    ├── modeling.ipynb               # Predictive modeling notebooks
plot images/
    └── ...                         # Saved visualizations and plots
script/
    └── ...                         # Python scripts for data processing or model training
data/
    ├── raw/                        # (Not included) Raw data downloads
    ├── processed/                  # Cleaned and processed datasets
requirements.txt                    # Python dependencies
README.md                           # Project documentation
```

> **Note:** Data files are not included due to licensing restrictions. Scripts and notebooks assume one have access to the required data sources.

## Methodology

1. **Data Collection**  
   - Historical stock prices from sources like Yahoo Finance (`yfinance`) or `pandas-datareader`.
   - News headlines/articles scraped or downloaded from financial news APIs.

2. **Exploratory Data Analysis**  
   - Statistical summary, correlation analysis, and visualization of stock prices.
   - Distribution analysis of sentiment scores.

3. **Sentiment Extraction**  
   - NLP libraries (e.g., `spaCy`, `TextBlob`, `NLTK`, `transformers`) are used to compute sentiment scores from text.
   - Sentiment time-series are aligned with stock price time-series.

4. **Feature Engineering**  
   - Merge of price features (returns, volatility, technical indicators) with sentiment features.
   - Handling of missing values, normalization, and date alignment.

5. **Modeling and Prediction**  
   - Baseline models: ARIMA, linear regression, etc.
   - Deep learning: LSTM, GRU with and without sentiment inputs.
   - Model evaluation: RMSE, MAE, directional accuracy, and visualization of predictions.

6. **Visualization and Interpretation**  
   - Graphs for feature relationships, model predictions, and error analysis.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/nuhaminae/Stock-Market-Price-Prediction-with-Sentiment-Analysis.git
    cd Stock-Market-Price-Prediction-with-Sentiment-Analysis
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare data**
   - Download or collect historical stock price data and relevant news/sentiment data.
   - Place data in the appropriate `data/raw/` directory.

### Usage

1. **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
2. **Open and run notebooks**
   - Start with `notebooks/historical_price_eda.ipynb` and `notebooks/news_eda.ipynb` for data exploration.
   - Proceed to `notebooks/modeling.ipynb` for predictive modeling.

3. **View plots**
   - Generated plots are saved in the `plot images/` directory for review.

## Requirements

The main dependencies (see `requirements.txt` for the full list):

- numpy, pandas, matplotlib, seaborn, plotly
- scikit-learn, scipy
- yfinance, pandas-datareader
- nltk, spacy, textblob, transformers (for NLP and sentiment)
- torch, torchvision (for deep learning)
- tqdm, joblib, holidays, ta (technical analysis), and more

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Results and Visualizations

- Key findings, model results, and sample visualizations are saved in the `plot images/` directory.
- For detailed results, refer to the output cells in each notebook.

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo, open issues, or submit pull requests.

**To contribute:**
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

---

> **Project Status:**  
This project is completed. See the [commit history](https://github.com/nuhaminae/Stock-Market-Price-Prediction-with-Sentiment-Analysis/commits?per_page=5&sort=updated) for changes.
