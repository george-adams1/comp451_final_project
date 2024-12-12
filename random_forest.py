import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os



# Reuse the prepare_data function from the previous files
def prepare_data(df, window=20):
    """Prepare features and target for training"""
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()

    # Create features (lagged returns and moving averages)
    for i in range(1, window + 1):
        df[f'Return_Lag_{i}'] = df['Return'].shift(i)
        df[f'MA_{i}'] = df['Close'].rolling(window=i).mean()

    # Create binary target (1 if price goes up, 0 if down)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    # Drop NaN values
    df = df.dropna()

    # Prepare features and target
    features = [col for col in df.columns if col.startswith(('Return_Lag_', 'MA_'))]
    X = df[features]
    y = df['Target']

    return X, y


# Load the combined data
data = pd.read_csv('stock_data/combined_stock_data.csv')

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Initialize results storage
results = []

# Train model for each ticker
for ticker in data['Ticker'].unique():
    print(f"\nTraining Random Forest for {ticker}")

    # Get data for this ticker
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('Date')

    # Prepare features and target
    X, y = prepare_data(ticker_data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results.append({
        'Ticker': ticker,
        'Model': 'Random Forest',
        'Accuracy': accuracy
    })

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Accuracy for {ticker}: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nTop 5 Important Features:")
    print(feature_importance.head())

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/random_forest_results.csv', index=False)
print("\nResults saved to 'results/random_forest_results.csv'")