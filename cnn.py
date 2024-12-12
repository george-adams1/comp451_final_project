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
    """Custom Dataset for stock data with CNN-compatible format"""

    def __init__(self, X, y, sequence_length=20):
        # Reshape X to (batch_size, 1, sequence_length, features)
        # 1 is the channel dimension
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockCNN(nn.Module):
    """CNN for stock prediction"""

    def __init__(self, input_channels=1, sequence_length=20, n_features=11):
        super(StockCNN, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Calculate size after convolutions and pooling
        self.flatten_size = self._get_flatten_size(sequence_length, n_features)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_flatten_size(self, sequence_length, n_features):
        # Helper method to calculate flatten size
        with torch.no_grad():
            x = torch.randn(1, 1, sequence_length, n_features)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            return x.numel()

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x


def prepare_data(df, sequence_length=20):
    """Prepare features and target with sequence consideration"""
    df = df.copy()

    # Convert price and volume columns to numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate features (same as before)
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum20'] = df['Close'] / df['Close'].shift(20) - 1
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['High_Low'] = df['High'] - df['Low']
    df['Close_Open'] = df['Close'] - df['Open']

    # Target (1 if price goes up)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'Daily_Return', 'MA5', 'MA20', 'MA50',
        'Volume_MA5', 'Volume_MA20',
        'Momentum5', 'Momentum20',
        'Volatility', 'High_Low', 'Close_Open'
    ]

    # Remove rows with NaN values
    df = df.dropna()

    # Create sequences
    sequences = []
    targets = []

    for i in range(len(df) - sequence_length):
        seq = df[feature_cols].iloc[i:i + sequence_length].values
        target = df['Target'].iloc[i + sequence_length - 1]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """Train the CNN model"""
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

            # Prepare sequences and target
            sequence_length = 20  # You can adjust this
            X, y = prepare_data(stock_data, sequence_length)
            print(f"Processed data shape: X={X.shape}, y={y.shape}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            # Reshape to 2D for scaling
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])

            X_train_scaled_2d = scaler.fit_transform(X_train_2d)
            X_test_scaled_2d = scaler.transform(X_test_2d)

            # Reshape back to 3D
            X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
            X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

            # Create data loaders
            train_dataset = StockDataset(X_train_scaled, y_train)
            test_dataset = StockDataset(X_test_scaled, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize model
            model = StockCNN(
                input_channels=1,
                sequence_length=sequence_length,
                n_features=X.shape[-1]
            ).to(device)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train model
            print(f"\nTraining CNN model for {ticker}...")
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
                'Date': stock_data['Date'].iloc[sequence_length:].iloc[X_test.index],
                'Actual': y_test,
                'Predicted': test_preds,
                'Probability': test_probs
            })
            predictions_df.to_csv(f'results/{ticker}_cnn_predictions.csv', index=False)

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv('results/cnn_results.csv', index=False)

    print("\nFinal Results Summary:")
    print("=====================")
    print(results_df)
    print(f"\nAverage Accuracy: {results_df['Accuracy'].mean():.4f}")
else:
    print("\nNo results to save - all processing failed")