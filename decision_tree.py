import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
from utils import prepare_data


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

            # Quick data validation
            print("\nSample of raw data:")
            print(stock_data[['Date', 'Close', 'Volume']].head())

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
            model = DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=20,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

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

            # Feature importance
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\nTop 5 Important Features:")
            print(importance_df.head())

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            print("Data types:")
            print(stock_data.dtypes)
            continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv('results/decision_tree_results.csv', index=False)

    print("\nFinal Results Summary:")
    print("=====================")
    print(results_df)
    print(f"\nAverage Accuracy: {results_df['Accuracy'].mean():.4f}")
    print("\nResults saved to 'results/decision_tree_results.csv'")
else:
    print("\nNo results to save - all processing failed")