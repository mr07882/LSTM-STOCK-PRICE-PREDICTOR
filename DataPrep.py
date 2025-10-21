import numpy as np            #FOR NUMERICAL OPERATIONS
 
'''
Pipeline 2: Data Preparation
Now that we have the raw data, the next step is to prepare it for the 
LSTM model. Neural networks work best when numbers are small and 
consistent, so we first normalize the data: this means we scale all 
prices so that they’re roughly between -1 and +1. --->NORMALIZATION

Then, we create input–output pairs using a sliding window approach:
we take the prices from the past 60 days (that’s one “window”) 
and use those as the input to predict the price on the next day 
(the 61st day). This helps the model learn patterns from history.

The whole dataset is then split into training data (80%) and 
validation data (20%). Training data teaches the model,
and validation data tests how well it learned.

Finally, we convert these arrays into PyTorch datasets — 
a special format that the PyTorch library can easily work with. 
We wrap them into DataLoaders, which automatically feed small 
batches of data to the model during training.
'''

class Normalizer():
    def __init__(self):
        self.Meo = None
        self.SD = None

    def FitTransformation(self, x):
        self.Meo = np.mean(x, axis=(0), keepdims=True)
        self.SD = np.std(x, axis=(0), keepdims=True)
        NormalizedX = (x - self.Meo)/self.SD
        return NormalizedX

    def InverseTransformation(self, x):
        return (x*self.SD) + self.Meo 


def PrepDataX(x, PredictionCycle):
    n_row = x.shape[0] - PredictionCycle + 1
    Output = np.lib.stride_tricks.as_strided(x, shape=(n_row, PredictionCycle), strides=(x.strides[0], x.strides[0]))
    return Output[:-1], Output[-1]

def PrepDataY(x, PredictionCycle):
    Output = x[PredictionCycle:]
    return Output

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def PlotTrainTestData(DataDate, TrainingData_Output, TestingData_Output, Scalar, NumDataPoints, SplitIndex, Config):
    
    print("Preparing data for plotting...")
    to_plot_TrainingData_Output = np.zeros(NumDataPoints)
    to_plot_TestingData_Output = np.zeros(NumDataPoints)

    cycle = Config["Data"]["PredictionCycle"]

    # Reverse normalization for plotting
    to_plot_TrainingData_Output[cycle:SplitIndex + cycle] = Scalar.InverseTransformation(TrainingData_Output)
    to_plot_TestingData_Output[SplitIndex + cycle:] = Scalar.InverseTransformation(TestingData_Output)

    # Replace zeros with None to avoid gaps
    to_plot_TrainingData_Output = np.where(to_plot_TrainingData_Output == 0, None, to_plot_TrainingData_Output)
    to_plot_TestingData_Output = np.where(to_plot_TestingData_Output == 0, None, to_plot_TestingData_Output)

    # Plot
    print("Plotting training and validation data...")
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))

    plt.plot(DataDate, to_plot_TrainingData_Output, label="Prices (train)", color=Config["Plots"]["color_train"])
    plt.plot(DataDate, to_plot_TestingData_Output, label="Prices (validation)", color=Config["Plots"]["color_val"])

    xticks = [
        DataDate[i] if (
            (i % Config["Plots"]["X-Interval"] == 0 and (NumDataPoints - i) > Config["Plots"]["X-Interval"])
            or i == NumDataPoints - 1
        ) else None for i in range(NumDataPoints)
    ]
    x = np.arange(0, len(xticks))

    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + Config["AlphaVantage"]["Stock"] + " - showing training and validation data")
    plt.grid(True, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()



def SplitData(data_x, data_y, SplitIndex):
    TrainingData_Input = data_x[:SplitIndex]
    TestingData_Input = data_x[SplitIndex:]
    TrainingData_Output = data_y[:SplitIndex]
    TestingData_Output = data_y[SplitIndex:]
     
    return TrainingData_Input, TestingData_Input, TrainingData_Output, TestingData_Output



