import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
from utils import prepare_data


# def prepare_data(df):
#     """
#     Prepare features and target for training from individual stock data
#     """
#     df = df.copy()
#
#     # Convert price and volume columns to numeric
#     numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     # Calculate daily returns
#     df['Daily_Return'] = df['Close'].pct_change()
#
#     # Technical indicators
#     # Moving averages
#     df['MA5'] = df['Close'].rolling(window=5).mean()
#     df['MA20'] = df['Close'].rolling(window=20).mean()
#     df['MA50'] = df['Close'].rolling(window=50).mean()
#
#     # Volume indicators
#     df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
#     df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
#
#     # Momentum
#     df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1
#     df['Momentum20'] = df['Close'] / df['Close'].shift(20) - 1
#
#     # Volatility
#     df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
#
#     # Price differences
#     df['High_Low'] = df['High'] - df['Low']
#     df['Close_Open'] = df['Close'] - df['Open']
#
#     # Target: 1 if tomorrow's price is higher than today
#     df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
#
#     # Select features
#     feature_cols = [
#         'Daily_Return', 'MA5', 'MA20', 'MA50',
#         'Volume_MA5', 'Volume_MA20',
#         'Momentum5', 'Momentum20',
#         'Volatility', 'High_Low', 'Close_Open'
#     ]
#
#     # Remove rows with NaN values
#     df = df.dropna()
#
#     return df[feature_cols], df['Target']


# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Initialize results storage
results = []

# Process each stock file in the stock_data directory
for filename in os.listdir('stock_data'):
    if '_data.csv' in filename:  # Only process individual stock files
        ticker = filename.replace('_data.csv', '')
        print(f"\nProcessing {ticker}")

        try:
            # Load data
            data_path = os.path.join('stock_data', filename)
            stock_data = pd.read_csv(data_path)
            print(f"Loaded data shape: {stock_data.shape}")

            # Convert date
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            # Prepare features and target
            X, y = prepare_data(stock_data)
            print(f"Processed data shape: X={X.shape}, y={y.shape}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                shuffle=False  # Keep chronological order
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,  # Increase max iterations for convergence
                C=0.1  # Increase regularization
            )
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate probabilities
            y_prob = model.predict_proba(X_test_scaled)

            # Store results
            results.append({
                'Ticker': ticker,
                'Accuracy': accuracy,
                'Train_Size': len(X_train),
                'Test_Size': len(X_test)
            })

            # Print detailed results
            print(f"\nResults for {ticker}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Feature coefficients (similar to importance in decision trees)
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=False)

            print("\nTop 5 Most Influential Features (by absolute coefficient):")
            print(coef_df.reindex(coef_df.abs().sort_values('Coefficient', ascending=False).index).head())

            # Save model predictions and probabilities
            predictions_df = pd.DataFrame({
                'Date': stock_data['Date'].iloc[X_test.index],
                'Actual': y_test,
                'Predicted': y_pred,
                'Prob_Down': y_prob[:, 0],
                'Prob_Up': y_prob[:, 1]
            })
            predictions_df.to_csv(f'results/{ticker}_predictions.csv', index=False)

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            print("Data types:")
            print(stock_data.dtypes)
            continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv('results/logistic_regression_results.csv', index=False)

    print("\nFinal Results Summary:")
    print("=====================")
    print(results_df)
    print(f"\nAverage Accuracy: {results_df['Accuracy'].mean():.4f}")
    print("\nResults saved to:")
    print("- Overall results: results/logistic_regression_results.csv")
    print("- Individual predictions: results/TICKER_predictions.csv")
else:
    print("\nNo results to save - all processing failed")
