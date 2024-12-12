import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os


def download_and_process_stock_data(tickers, start_date, end_date):
    """
    Download and process stock data with proper handling of OHLCV data

    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    """
    # Create output directory
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    all_stocks_data = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")

        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            # Basic features
            df['Daily_Return'] = df['Close'].pct_change()

            # Technical indicators
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            # Volatility
            df['Daily_Volatility'] = df['Daily_Return'].rolling(window=20).std()

            # Volume features
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

            # Price momentum
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

            # Target variable (1 if next day's price is higher, 0 if lower)
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Add ticker column
            df['Ticker'] = ticker

            # Save individual stock data
            df.to_csv(f'processed_data/{ticker}_processed.csv')
            all_stocks_data[ticker] = df

            print(f"Successfully processed {ticker}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print("Features created:", list(df.columns))
            print("\n")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    # Combine all data
    combined_data = pd.concat(all_stocks_data.values(), axis=0)
    combined_data.to_csv('processed_data/all_stocks_processed.csv')

    return all_stocks_data


def print_data_summary(all_stocks_data):
    """Print summary statistics for all processed stock data"""
    for ticker, df in all_stocks_data.items():
        print(f"\nSummary for {ticker}:")
        print("-------------------")
        print(f"Total days: {len(df)}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print("\nBasic statistics for Close price:")
        print(df['Close'].describe())
        print("\nTarget distribution:")
        print(df['Target'].value_counts(normalize=True))


# List of stock tickers
tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'JNJ', 'PG']

# Define the time period
start_date = '2019-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Download and process data
all_stocks_data = download_and_process_stock_data(tickers, start_date, end_date)

# Print summary
print("\nData Processing Summary")
print("=====================")
print_data_summary(all_stocks_data)