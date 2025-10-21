import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

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


def TrainAndEvaluate(model, dataset_train, dataset_val, scaler, data_y_val, data_x_unseen, data_date, data_close_price, num_data_points, split_index, config):
    device = torch.device(config["Training"]["Device"]) if isinstance(config["Training"]["Device"], str) else config["Training"]["Device"]

    train_dataloader = DataLoader(dataset_train, batch_size=config["Training"]["BatchSize"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["Training"]["BatchSize"], shuffle=True)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["Training"]["LearningRate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["Training"]["StepSize"], gamma=0.1)

    for epoch in range(config["Training"]["EPOCHS"]):
        loss_train, lr_train = run_epoch(train_dataloader, model, optimizer, criterion, scheduler, config, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader, model, optimizer, criterion, scheduler, config)
        scheduler.step()

        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
              .format(epoch + 1, config["Training"]["EPOCHS"], loss_train, loss_val, lr_train))

    # re-initialize dataloader so the data doesn't shuffle for plotting/prediction
    train_dataloader = DataLoader(dataset_train, batch_size=config["Training"]["BatchSize"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=config["Training"]["BatchSize"], shuffle=False)

    model.eval()

    # predict on the training data
    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data
    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # prepare data for plotting
    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[config["Data"]["PredictionCycle"]:split_index+config["Data"]["PredictionCycle"]] = scaler.InverseTransformation(predicted_train)
    to_plot_data_y_val_pred[split_index+config["Data"]["PredictionCycle"]:] = scaler.InverseTransformation(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # plots - full
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=config["Plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["Plots"]["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["Plots"]["color_pred_val"])
    plt.title("Compare predicted prices to actual prices")
    xticks = [data_date[i] if ((i%config["Plots"]["X-Interval"]==0 and (num_data_points-i) > config["Plots"]["X-Interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # zoomed in plot for validation
    to_plot_data_y_val_subset = scaler.InverseTransformation(data_y_val)
    to_plot_predicted_val = scaler.InverseTransformation(predicted_val)

    true_prices = scaler.InverseTransformation(data_y_val)
    pred_prices = scaler.InverseTransformation(predicted_val)

    mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
    accuracy = 100 - mape

    print(f"\nRegression Accuracy (based on MAPE): {accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    to_plot_data_date = data_date[split_index+config["Data"]["PredictionCycle"]:]

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["Plots"]["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["Plots"]["color_pred_val"])
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [to_plot_data_date[i] if ((i%int(config["Plots"]["X-Interval"]/5)==0 and (len(to_plot_data_date)-i) > config["Plots"]["X-Interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))]
    xs = np.arange(0,len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    # predict the closing price of the next trading day
    model.eval()

    x = torch.tensor(data_x_unseen).float().to(device).unsqueeze(0).unsqueeze(2)
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()

    # prepare plots for last few days + tomorrow
    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range-1] = scaler.InverseTransformation(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = scaler.InverseTransformation(predicted_val)[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = scaler.InverseTransformation(prediction)

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    plot_date_test = data_date[-plot_range+1:]
    plot_date_test.append("tomorrow")

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["Plots"]["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["Plots"]["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["Plots"]["color_pred_test"])
    plt.title("Predicting the close price of the next trading day")
    plt.grid(True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))

    return model
