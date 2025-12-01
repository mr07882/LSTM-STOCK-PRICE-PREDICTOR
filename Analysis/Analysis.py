from ..DataGeneration import FetchData , GetStockFilePath
from ..Configuration import Config
from ..DataPrep import Normalizer , PrepDataX , PrepDataY, SplitData
from ..TimeSeries import TimeSeriesDataset
from ..LSTM_Model import LSTMModel, TrainModel , EvaluateModel 
from ..XGBoost_Model import TrainXGBModel , EvaluateXGBModel 
from ..N_BEATS_Model import TrainNBeatsModel , EvaluateNBeatsModel 
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def ALL_MODELS(stock_symbols: list):
    # Ensure the MODELS directory exists
    models_dir = "MODELS"
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    for stock_symbol in stock_symbols:
        print(f"Training models for stock: {stock_symbol}")

        # Fetch and prepare data
        filepath = GetStockFilePath(stock_symbol)
        DataDate, DataClosePrice, NumDataPoints, DateRange = FetchData(filepath, Config)

        Scalar = Normalizer()
        NormalizedDataClosePrice = Scalar.FitTransformation(DataClosePrice)
        data_x, data_x_unseen = PrepDataX(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"])
        data_y = PrepDataY(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"])

        SplitIndex = int(data_y.shape[0] * Config["Data"]["DataSplitRatio"])
        TrainingData_Input, TestingData_Input, TrainingData_Output, TestingData_Output = SplitData(data_x, data_y, SplitIndex)

        TrainingDataset = TimeSeriesDataset(TrainingData_Input, TrainingData_Output)
        TestingDataset = TimeSeriesDataset(TestingData_Input, TestingData_Output)

        # Train and evaluate LSTM model
        lstm_model = LSTMModel(input_size=Config["Model"]["InputSize"],
                               hidden_layer_size=Config["Model"]["LSTMSize"],
                               num_layers=Config["Model"]["NumOfLSTMLayers"],
                               output_size=1,
                               dropout=Config["Model"]["Dropout"])

        lstm_model = TrainModel(lstm_model, TrainingDataset, TestingDataset, Config)
        _, _, lstm_mape, lstm_accuracy = EvaluateModel(lstm_model, TrainingDataset, TestingDataset, Scalar, Config)
        lstm_path = os.path.join(models_dir, f"LSTM_{stock_symbol}.pth")
        #torch.save(lstm_model.state_dict(), lstm_path)
        #print(f"LSTM model saved to {lstm_path}")

        # Train and evaluate XGBoost model
        xgb_model = TrainXGBModel(TrainingData_Input, TrainingData_Output, TestingData_Input, TestingData_Output)
        xgb_training_preds, xgb_testing_preds, xgb_mape, xgb_accuracy = EvaluateXGBModel(
            xgb_model,
            TrainingData_Input,
            TestingData_Input,
            TestingData_Output,
            Scalar,
        )
        xgb_path = os.path.join(models_dir, f"XGBoost_{stock_symbol}.json")
        #xgb_model.save_model(xgb_path)
        #print(f"XGBoost model saved to {xgb_path}")

        # Train and evaluate N-BEATS model
        nbeats_model = TrainNBeatsModel(TrainingDataset, TestingDataset, Config)
        nbeats_train_preds, nbeats_test_preds, nbeats_mape, nbeats_accuracy = EvaluateNBeatsModel(
            nbeats_model, TrainingDataset, TestingDataset, Scalar
        )
        nbeats_path = os.path.join(models_dir, f"NBeats_{stock_symbol}.pth")
        #torch.save(nbeats_model.state_dict(), nbeats_path)
        #print(f"N-BEATS model saved to {nbeats_path}")

        # Store results
        results[stock_symbol] = {
            "LSTM": (lstm_accuracy, lstm_mape),
            "XGBoost": (xgb_accuracy, xgb_mape),
            "NBeats": (nbeats_accuracy, nbeats_mape)
        }

    return results

#UNCOMMENT TO RUN ALL MODELS AND SAVE RESULTS
# stock_list = ["AAPL", "ADBE", "AMZN", "ASML", "GOOG", "IBM", "INTC", "MCD", "META", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "SONY", "TSLA", "WMT"]
# all_model_results = ALL_MODELS(stock_symbols=stock_list)
# with open("all_model_results.pkl", "wb") as f:
#     pickle.dump(all_model_results, f)
# print("All Model Results:", all_model_results)



def Cycle_Analysis():
    cycles = [30, 60, 90, 120]
    cycle_results = {}

    for cycle in cycles:
        print(f"Running analysis for prediction cycle: {cycle}")
        Config["Data"]["PredictionCycle"] = cycle

        # Run ALL_MODELS without saving the models
        stock_list = ["AAPL", "ADBE", "AMZN", "ASML", "GOOG", "IBM", "INTC", "MCD", "META", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "SONY", "TSLA", "WMT"]
        results = ALL_MODELS(stock_list)

        # Store results for the current cycle
        cycle_results[cycle] = results

    return cycle_results

#UNCOMMENT TO RUN CYCLE ANALYSIS AND SAVE RESULTS
# cycle_results = Cycle_Analysis()
# with open("cycle_results.pkl", "wb") as f:
#      pickle.dump(cycle_results, f)
# print("All Cycle Results:", cycle_results)




with open("all_model_results.pkl", "rb") as f:
    results = pickle.load(f)

# Extract stock names and accuracies
stock_names = list(results.keys())
lstm_accuracies = [results[stock]['LSTM'][0] for stock in stock_names]
xgb_accuracies = [results[stock]['XGBoost'][0] for stock in stock_names]
nbeats_accuracies = [results[stock]['NBeats'][0] for stock in stock_names]

# Bar width and positions
bar_width = 0.25
x = np.arange(len(stock_names))

# Plot bars
plt.figure(figsize=(14, 7))
plt.bar(x - bar_width, lstm_accuracies, width=bar_width, label='LSTM')
plt.bar(x, xgb_accuracies, width=bar_width, label='XGBoost')
plt.bar(x + bar_width, nbeats_accuracies, width=bar_width, label='NBeats')

# Add labels and title
plt.xlabel('Stock Name', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison by Stock', fontsize=14)
plt.xticks(ticks=x, labels=stock_names, rotation=45, fontsize=10)
plt.legend()
plt.tight_layout()

# Show the plot
# Save the plot to a file
plt.savefig("Stock_Model_BarChart.png")

# Calculate average accuracy for each model
lstm_avg_accuracy = np.mean([results[stock]['LSTM'][0] for stock in results])
xgb_avg_accuracy = np.mean([results[stock]['XGBoost'][0] for stock in results])
nbeats_avg_accuracy = np.mean([results[stock]['NBeats'][0] for stock in results])

# Prepare data for the bar graph
models = ['LSTM', 'XGBoost', 'NBeats']
average_accuracies = [lstm_avg_accuracy, xgb_avg_accuracy, nbeats_avg_accuracy]

# Plot the bar graph
plt.figure(figsize=(8, 5))
bars = plt.bar(models, average_accuracies, color=['blue', 'orange', 'green'])

# Add labels and title
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.title('Average Accuracy of Each Model', fontsize=14)
plt.ylim(0, 100)

# Add average accuracy values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', fontsize=10)

# Save the plot to a file
plt.savefig("AverageModelAccuracy.png")


with open("cycle_results.pkl", "rb") as f:
    CycleResults = pickle.load(f)

# Extract average accuracy for each model across cycles
cycles = [30, 60, 90, 120]
models = ['LSTM', 'XGBoost', 'NBeats']

# Initialize a dictionary to store average accuracies for each model in each cycle
average_accuracies = {model: [] for model in models}

# Calculate average accuracies for each model in each cycle
for cycle in cycles:
    for model in models:
        accuracies = [CycleResults[cycle][stock][model][0] for stock in CycleResults[cycle]]
        avg_accuracy = np.mean(accuracies)
        average_accuracies[model].append(avg_accuracy)

# Plot the bar graph
bar_width = 0.2
x = np.arange(len(models))

plt.figure(figsize=(12, 6))

# Plot bars for each cycle
for i, cycle in enumerate(cycles):
    cycle_accuracies = [average_accuracies[model][i] for model in models]
    plt.bar(x + i * bar_width, cycle_accuracies, width=bar_width, label=f'Cycle {cycle}')

# Add labels, title, and legend
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.title('Average Accuracy of Models Across Prediction Cycles', fontsize=14)
plt.xticks(ticks=x + bar_width * 1.5, labels=models, fontsize=10)
plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig("Model_Average_Accuracy_Cycles.png")
plt.show()