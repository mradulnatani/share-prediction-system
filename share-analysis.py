import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return data

# Custom Dataset
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# CNN Model
class StockPricePredictorCNN(nn.Module):
    def __init__(self, input_size=1, num_filters=64, kernel_size=3, hidden_size=50, sequence_length=60):
        super(StockPricePredictorCNN, self).__init__()
        # Calculate the output size after convolutions and pooling
        conv_output_length = sequence_length
        for _ in range(2):  # Two conv layers with padding and max pooling
            conv_output_length = (conv_output_length - kernel_size + 2) // 2 + 1

        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Dynamically calculate the flattened size
        flattened_size = num_filters * conv_output_length
        
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Ensure the input is 3D: (batch_size, input_size, sequence_length)
        x = x.squeeze(-1).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    return scaled_data, scaler

# Prepare data loaders
def prepare_dataloaders(scaled_data, sequence_length=60, batch_size=32):
    dataset = StockDataset(scaled_data, sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            # Reshape and add channel dimension
            x_batch = x_batch.unsqueeze(-1)
            y_batch = y_batch.unsqueeze(-1)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Predict and analyze stock
def predict_and_analyze(model, data, scaler, sequence_length=60):
    model.eval()
    predictions = []
    all_data = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        for i in range(sequence_length, len(all_data)):
            input_seq = all_data[i - sequence_length:i].unsqueeze(0).unsqueeze(-1)
            pred = model(input_seq)
            predictions.append(pred.item())

    # Inverse transform predictions and original data
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_prices = scaler.inverse_transform(data)

    # Calculate key metrics
    last_actual_price = actual_prices[-1][0]
    predicted_next_price = predicted_prices[-1][0]
    
    # Price change analysis
    price_change_percent = ((predicted_next_price - last_actual_price) / last_actual_price) * 100
    
    # Generate recommendation
    if price_change_percent > 5:
        recommendation = "STRONG BUY"
        recommendation_reason = f"Predicted {price_change_percent:.2f}% price increase"
    elif price_change_percent > 2:
        recommendation = "BUY"
        recommendation_reason = f"Predicted {price_change_percent:.2f}% price increase"
    elif price_change_percent > -2:
        recommendation = "HOLD"
        recommendation_reason = f"Minimal price change predicted ({price_change_percent:.2f}%)"
    elif price_change_percent > -5:
        recommendation = "SELL"
        recommendation_reason = f"Predicted {price_change_percent:.2f}% price decrease"
    else:
        recommendation = "STRONG SELL"
        recommendation_reason = f"Predicted {price_change_percent:.2f}% price decrease"

    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Actual prices subplot
    plt.subplot(2, 1, 1)
    plt.plot(actual_prices, label="Actual Prices", color='blue')
    plt.title("Actual Stock Prices")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()

    # Predictions subplot
    plt.subplot(2, 1, 2)
    plt.plot(range(sequence_length, len(predicted_prices) + sequence_length), 
             predicted_prices, label="Predicted Prices", color='red')
    plt.title("Predicted Stock Prices")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print analysis
    print("\n--- Stock Analysis ---")
    print(f"Current Price: ${last_actual_price:.2f}")
    print(f"Predicted Next Price: ${predicted_next_price:.2f}")
    print(f"Price Change: {price_change_percent:.2f}%")
    print(f"Recommendation: {recommendation}")
    print(f"Reason: {recommendation_reason}")

    return {
        'current_price': last_actual_price,
        'predicted_price': predicted_next_price,
        'price_change_percent': price_change_percent,
        'recommendation': recommendation,
        'recommendation_reason': recommendation_reason
    }

# Main function
def main():
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    print("Fetching stock data...")
    data = fetch_stock_data(ticker, start_date, end_date)

    print("Preprocessing data...")
    scaled_data, scaler = preprocess_data(data)

    sequence_length = 60
    batch_size = 32

    print("Preparing data loaders...")
    train_loader, val_loader = prepare_dataloaders(scaled_data, sequence_length, batch_size)

    print("Initializing model...")
    model = StockPricePredictorCNN(sequence_length=sequence_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    print("Generating predictions and analysis...")
    analysis = predict_and_analyze(model, scaled_data, scaler, sequence_length)

if __name__ == "__main__":
    main()