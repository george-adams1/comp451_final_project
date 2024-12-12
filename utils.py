import pandas as pd

def prepare_data(df):
    """
    Prepare features and target for training from individual stock data
    """
    df = df.copy()

    # Convert price and volume columns to numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Technical indicators
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # Momentum
    df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum20'] = df['Close'] / df['Close'].shift(20) - 1

    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    # Price differences
    df['High_Low'] = df['High'] - df['Low']
    df['Close_Open'] = df['Close'] - df['Open']

    # Target: 1 if tomorrow's price is higher than today
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Select features
    feature_cols = [
        'Daily_Return', 'MA5', 'MA20', 'MA50',
        'Volume_MA5', 'Volume_MA20',
        'Momentum5', 'Momentum20',
        'Volatility', 'High_Low', 'Close_Open'
    ]

    # Remove rows with NaN values
    df = df.dropna()

    return df[feature_cols], df['Target']