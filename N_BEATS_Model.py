import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_percentage_error



class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_layer_units=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_units)
        self.fc2 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc3 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.fc4 = nn.Linear(hidden_layer_units, hidden_layer_units)
        self.theta_layer = nn.Linear(hidden_layer_units, theta_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        theta = self.theta_layer(x)
        return theta


class NBeatsModel(nn.Module):
    def __init__(self, input_size=60, output_size=1, hidden_units=128, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, hidden_units) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        x = x.squeeze(-1)  # -> [batch, seq_len]
        forecast = 0
        for block in self.blocks:
            forecast += block(x)
        return forecast  # shape [batch, 1]


# ============================================================
# TRAINING FUNCTION
# ============================================================

def TrainNBeatsModel(TrainingDataset, TestingDataset, Config):
    """
    Trains an N-BEATS model on the provided time-series dataset.
    """
    print("PROCESS: Training N-BEATS model................")
    device = torch.device(Config["Training"]["Device"]) if isinstance(Config["Training"]["Device"], str) else Config["Training"]["Device"]

    train_loader = DataLoader(TrainingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=True)
    test_loader = DataLoader(TestingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=False)

    model = NBeatsModel(
        input_size=Config["Data"]["PredictionCycle"],
        output_size=1,
        hidden_units=128,
        num_blocks=4
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config["Training"]["LearningRate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config["Training"]["StepSize"], gamma=0.1)

    print("Training N-BEATS model...")
    for epoch in range(Config["Training"]["EPOCHS"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x).squeeze(-1)  # [batch]
            y = y.view(-1)                 # [batch]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{Config['Training']['EPOCHS']}], Loss: {total_loss/len(train_loader):.6f}")

    print("Training complete.")
    print("CHECKPOINT 1: Training Completed.")
    return model


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def EvaluateNBeatsModel(model, TrainingDataset, TestingDataset, Scalar):
    print("PROCESS: Evaluating N-BEATS model on testing data................")
    device = next(model.parameters()).device
    train_loader = DataLoader(TrainingDataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(TestingDataset, batch_size=64, shuffle=False)

    model.eval()
    train_preds, test_preds = [], []
    y_train_true, y_test_true = [], []

    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(-1).cpu().numpy().flatten()
            train_preds.extend(out.tolist())
            y_train_true.extend(y.cpu().numpy().flatten().tolist())

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(-1).cpu().numpy().flatten()
            test_preds.extend(out.tolist())
            y_test_true.extend(y.cpu().numpy().flatten().tolist())

    # Convert to numpy arrays
    train_preds = np.array(train_preds)
    test_preds = np.array(test_preds)
    y_train_true = np.array(y_train_true)
    y_test_true = np.array(y_test_true)

    # Inverse transform for metrics
    true_prices = Scalar.InverseTransformation(y_test_true)
    pred_prices = Scalar.InverseTransformation(test_preds)

    MAPE = mean_absolute_percentage_error(true_prices, pred_prices) * 100
    Accuracy = 100 - MAPE

    print(f"\nRegression Accuracy (based on MAPE): {Accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE:.2f}%")

    print("CHECKPOINT 2: Evaluation Completed.")
    return train_preds, test_preds, MAPE, Accuracy


# ============================================================
# PREDICTION FUNCTION
# ============================================================

# def PredictNextDayNBeats(model, LastWindow, Scalar):
#     """
#     Predicts next day's closing price using the trained N-BEATS model.
#     """
#     print("PROCESS: Predicting next day's closing price using N-BEATS................")

    
#     device = next(model.parameters()).device
#     model.eval()

#     # Ensure proper shape: [1, seq_len, 1]
#     x = torch.tensor(LastWindow).float().unsqueeze(0).unsqueeze(2).to(device)
#     with torch.no_grad():
#         pred = model(x).squeeze(-1).cpu().numpy()

#     # NextDayPrice = Scalar.InverseTransformation(pred)
#     print(f"Predicted next-day closing price: {float(pred[0]):.2f}")
#     print("CHECKPOINT 3: Prediction Completed.")
#     return float(pred[0])

def PredictNextDayNBeats(model, LastWindow):
    """
    Predicts the next day's closing price using the trained N-BEATS model.
    LastWindow should be a 1D array (or 2D with a single row) of length window_size.
    """
    print("PROCESS: Predicting next day's closing price using N-BEATS................")

    # Convert to numpy array
    arr = np.array(LastWindow)
    if arr.ndim == 2:
        arr = arr[-1]  # handle shape (1, seq_len)
    LastWindow = arr.reshape(1, -1, 1)  # ensure shape [1, seq_len, 1]

    # Convert to torch tensor and move to model's device
    device = next(model.parameters()).device
    x = torch.tensor(LastWindow).float().to(device)

    # Make prediction
    with torch.no_grad():
        Prediction = model(x).cpu().numpy()

    print("CHECKPOINT 3: Prediction Completed.")
    return Prediction

