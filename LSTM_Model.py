import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from Plotter import Plot

'''
THE LSTM MODEL WILL LOOK AT A SHORT SEQUENCE OF PAST STOCK PRICES 
(SAY 60 DAYS WHICH WE SET IN CONFIGURATION FILE), PROCESSES EACH VALUE
INTO A RICHER VECTOR AND READS THE WHOLE SEQUENCE WITH AN LSTM 
AND THEN USES THE FINAL HIDDEN STATES TO PREDICT THE NEXT DAY'S 
CLOSING PRICE.
'''

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        # using keyword args compatible with older/newer pytorch
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, model, optimizer, criterion, scheduler, config, is_training=False):
    epoch_loss = 0

    device = torch.device(config["Training"]["Device"]) if isinstance(config["Training"]["Device"], str) else config["Training"]["Device"]

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0] if scheduler is not None else None

    return epoch_loss, lr



import numpy as np

def ModelPerformance(TestingData_Output, TestingPredictions, Scalar):
    """
    Evaluates model performance using MAPE and Accuracy.

    Parameters
    ----------
    TestingData_Output : np.ndarray
        The actual target values from validation/testing set (normalized).
    TestingPredictions : np.ndarray
        The model's predicted values for validation/testing set (normalized).
    Scalar : object
        The scaler/normalizer used for inverse transformation.

    Returns
    -------
    tuple : (mape, accuracy)
        MAPE: Mean Absolute Percentage Error (%)
        Accuracy: Regression accuracy (%)
    """

    # Reverse normalization for accurate comparison
    true_prices = Scalar.InverseTransformation(TestingData_Output)
    pred_prices = Scalar.InverseTransformation(TestingPredictions)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100

    # Calculate Accuracy (%)
    accuracy = 100 - mape

    print(f"\nRegression Accuracy (based on MAPE): {accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return mape, accuracy


def TrainModel(Model, TrainingDataset, TestingDataset, Config):
    Device = torch.device(Config["Training"]["Device"]) if isinstance(Config["Training"]["Device"], str) else Config["Training"]["Device"]

    TrainingDataLoader = DataLoader(TrainingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=True)
    TestingDataLoader = DataLoader(TestingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=True)

    Model = Model.to(Device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Model.parameters(), lr=Config["Training"]["LearningRate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config["Training"]["StepSize"], gamma=0.1)

    for epoch in range(Config["Training"]["EPOCHS"]):
        TrainingLoss, LR_Train = run_epoch(TrainingDataLoader, Model, optimizer, criterion, scheduler, Config, is_training=True)
        TestingLoss, LR_Test = run_epoch(TestingDataLoader, Model, optimizer, criterion, scheduler, Config)
        scheduler.step()
        print(f"Epoch[{epoch+1}/{Config['Training']['EPOCHS']}] | Train Loss:{TrainingLoss:.6f} | Test Loss:{TestingLoss:.6f} | LR:{LR_Train:.6f}")

    return Model

def EvaluateModel(Model, TrainingDataset, TestingDataset, Scalar, Config):
    Device = torch.device(Config["Training"]["Device"]) if isinstance(Config["Training"]["Device"], str) else Config["Training"]["Device"]

    TrainingLoader = DataLoader(TrainingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=False)
    TestingLoader = DataLoader(TestingDataset, batch_size=Config["Training"]["BatchSize"], shuffle=False)

    Model.eval()
    TrainingPredictions, TestingPredictions = np.array([]), np.array([])

    for x, _ in TrainingLoader:
        x = x.to(Device)
        out = Model(x).cpu().detach().numpy()
        TrainingPredictions = np.concatenate((TrainingPredictions, out))

    for x, _ in TestingLoader:
        x = x.to(Device)
        out = Model(x).cpu().detach().numpy()
        TestingPredictions = np.concatenate((TestingPredictions, out))

    # Evaluate MAPE/Accuracy
    mape, accuracy = ModelPerformance(TestingDataset.y, TestingPredictions, Scalar)

    return TrainingPredictions, TestingPredictions, mape, accuracy

def PredictNextDay(Model, TestingData_Input, Config):
    Device = torch.device(Config["Training"]["Device"]) if isinstance(Config["Training"]["Device"], str) else Config["Training"]["Device"]

    Model.eval()

    # Accept either a single window (shape: (seq_len,)) or an array of windows (shape: (num_windows, seq_len)).
    arr = np.array(TestingData_Input)

    if arr.ndim == 2:
        # If a matrix was passed, use the last window as the sample to predict 'next day'.
        sample = arr[-1]
    elif arr.ndim == 1:
        sample = arr
    else:
        raise ValueError("TestingData_Input must be a 1D window or 2D array of windows")

    # Build tensor with shape (batch=1, seq_len, features=1)
    x = torch.tensor(sample).float().to(Device).unsqueeze(0).unsqueeze(2)

    with torch.no_grad():
        Prediction = Model(x)
        Prediction = Prediction.cpu().detach().numpy()

    # Inverse transform to price scale
    # try:
    #     NextDayPrice = Scalar.InverseTransformation(Prediction)
    # except Exception:
    #     NextDayPrice = Scalar.inverse_transform(Prediction)

    # print(f"Predicted next-day closing price: {NextDayPrice[0]:.2f}")

    # return float(NextDayPrice[0])

    return Prediction

   
    

