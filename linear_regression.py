import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import os


def prepare_data(df):
    """
    Prepare features and target with both continuous and binary targets
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

    # Targets:
    # 1. Continuous: Next day's price change in dollars
    df['Price_Change'] = df['Close'].shift(-1) - df['Close']
    # 2. Binary: Whether price goes up (1) or down (0)
    df['Target'] = (df['Price_Change'] > 0).astype(int)

    # Select features
    feature_cols = [
        'Daily_Return', 'MA5', 'MA20', 'MA50',
        'Volume_MA5', 'Volume_MA20',
        'Momentum5', 'Momentum20',
        'Volatility', 'High_Low', 'Close_Open'
    ]

    # Remove rows with NaN values
    df = df.dropna()

    return df[feature_cols], df['Price_Change'], df['Target']


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

            # Prepare features and targets
            X, y_cont, y_binary = prepare_data(stock_data)
            print(f"Processed data shape: X={X.shape}")

            # Split data
            X_train, X_test, y_cont_train, y_cont_test, y_binary_train, y_binary_test = train_test_split(
                X, y_cont, y_binary,
                test_size=0.2,
                shuffle=False  # Keep chronological order
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_cont_train)

            # Make predictions
            y_pred_cont = model.predict(X_test_scaled)
            # Convert continuous predictions to binary (1 if predicted change > 0)
            y_pred_binary = (y_pred_cont > 0).astype(int)

            # Calculate metrics
            mse = mean_squared_error(y_cont_test, y_pred_cont)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_cont_test, y_pred_cont)
            accuracy = accuracy_score(y_binary_test, y_pred_binary)

            # Store results
            results.append({
                'Ticker': ticker,
                'RMSE': rmse,
                'R2': r2,
                'Classification_Accuracy': accuracy,
                'Train_Size': len(X_train),
                'Test_Size': len(X_test)
            })

            # Print detailed results
            print(f"\nResults for {ticker}:")
            print(f"RMSE: ${rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
            print(f"Classification Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_binary_test, y_pred_binary))

            # Feature coefficients
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=False)

            print("\nTop 5 Most Influential Features (by absolute coefficient):")
            print(coef_df.reindex(coef_df.abs().sort_values('Coefficient', ascending=False).index).head())

            # Save predictions
            predictions_df = pd.DataFrame({
                'Date': stock_data['Date'].iloc[X_test.index],
                'Actual_Change': y_cont_test,
                'Predicted_Change': y_pred_cont,
                'Actual_Direction': y_binary_test,
                'Predicted_Direction': y_pred_binary,
                'Correct_Prediction': y_binary_test == y_pred_binary
            })
            predictions_df.to_csv(f'results/{ticker}_linear_predictions.csv', index=False)

            # Additional analysis: Prediction confidence vs accuracy
            predictions_df['Prediction_Magnitude'] = abs(y_pred_cont)
            predictions_df['Magnitude_Quintile'] = pd.qcut(predictions_df['Prediction_Magnitude'], 5,
                                                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

            print("\nAccuracy by Prediction Magnitude:")
            accuracy_by_magnitude = predictions_df.groupby('Magnitude_Quintile')['Correct_Prediction'].mean()
            print(accuracy_by_magnitude)

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            print("Data types:")
            print(stock_data.dtypes)
            continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Classification_Accuracy', ascending=False)
    results_df.to_csv('results/linear_regression_results.csv', index=False)

    print("\nFinal Results Summary:")
    print("=====================")
    print(results_df)
    print(f"\nAverage Classification Accuracy: {results_df['Classification_Accuracy'].mean():.4f}")
    print(f"Average R2: {results_df['R2'].mean():.4f}")
    print(f"Average RMSE: ${results_df['RMSE'].mean():.4f}")
    print("\nResults saved to:")
    print("- Overall results: results/linear_regression_results.csv")
    print("- Individual predictions: results/TICKER_linear_predictions.csv")
else:
    print("\nNo results to save - all processing failed")