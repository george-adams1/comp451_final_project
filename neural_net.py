import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """Custom Dataset for stock data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockPredictor(nn.Module):
    """Neural Network for stock prediction"""

    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def prepare_data(df):
    """Prepare features and target"""
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

    # Target (1 if price goes up)
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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """Train the neural network"""
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions = (outputs > 0.5).float().cpu().numpy()
                val_preds.extend(predictions)
                val_true.extend(batch_y.numpy())

        val_accuracy = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Training Loss: {train_loss / len(train_loader):.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')

    return train_losses, val_accuracies


# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Initialize results storage
results = []

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Process each stock file
for filename in os.listdir('stock_data'):
    if '_data.csv' in filename:
        ticker = filename.replace('_data.csv', '')
        print(f"\nProcessing {ticker}")

        try:
            # Load data
            data_path = os.path.join('stock_data', filename)
            stock_data = pd.read_csv(data_path)
            print(f"Loaded data shape: {stock_data.shape}")

            # Prepare features and target
            X, y = prepare_data(stock_data)
            print(f"Processed data shape: X={X.shape}, y={y.shape}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create data loaders
            train_dataset = StockDataset(X_train_scaled, y_train.values)
            test_dataset = StockDataset(X_test_scaled, y_test.values)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize model
            model = StockPredictor(input_size=X.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train model
            print(f"\nTraining model for {ticker}...")
            train_losses, val_accuracies = train_model(
                model, train_loader, test_loader, criterion, optimizer,
                num_epochs=50, device=device
            )

            # Final evaluation
            model.eval()
            test_preds = []
            test_probs = []

            with torch.no_grad():
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    probs = outputs.cpu().numpy()
                    predictions = (outputs > 0.5).float().cpu().numpy()
                    test_preds.extend(predictions)
                    test_probs.extend(probs)

            # Calculate metrics
            accuracy = accuracy_score(y_test, test_preds)

            # Store results
            results.append({
                'Ticker': ticker,
                'Accuracy': accuracy,
                'Train_Size': len(X_train),
                'Test_Size': len(X_test)
            })

            # Print results
            print(f"\nResults for {ticker}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, test_preds))

            # Save predictions
            predictions_df = pd.DataFrame({
                'Date': stock_data['Date'].iloc[X_test.index],
                'Actual': y_test,
                'Predicted': test_preds,
                'Probability': test_probs
            })
            predictions_df.to_csv(f'results/{ticker}_nn_predictions.csv', index=False)

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv('results/neural_network_results.csv', index=False)

    print("\nFinal Results Summary:")
    print("=====================")
    print(results_df)
    print(f"\nAverage Accuracy: {results_df['Accuracy'].mean():.4f}")
else:
    print("\nNo results to save - all processing failed")