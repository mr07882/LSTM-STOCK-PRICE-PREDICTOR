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

def SplitData(data_x, data_y, SplitIndex):
    TrainingData_Input = data_x[:SplitIndex]
    TestingData_Input = data_x[SplitIndex:]
    TrainingData_Output = data_y[:SplitIndex]
    TestingData_Output = data_y[SplitIndex:]
     
    return TrainingData_Input, TestingData_Input, TrainingData_Output, TestingData_Output



