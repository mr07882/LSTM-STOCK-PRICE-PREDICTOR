import numpy as np
from torch.utils.data import Dataset                 #FOR DATA HANDLING 


'''
THIS CLASS HAS A TINY HELPER WHICH WRAPS OUR DATASETS INTO A FORMAT THAT
PY TORCH LIBRARY (FOR DEEP LEARNING) CAN UNDERSTAND. THE RESULTING DATASET 
WILL HAVE ITEMS LIKE (x, y) WHERE x IS THE INPUT AND y IS THE OUTPUT.

'''

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):  # x: input dataset, y: output dataset
        x = np.expand_dims(x,
                           2)  # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


