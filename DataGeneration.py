#DEPENDENCIES
from alpha_vantage.timeseries import TimeSeries  #FOR DATA DOWNLOAD
import matplotlib.pyplot as plt                  #FOR PLOTTING
from matplotlib.pyplot import figure  
import numpy as np                               #FOR NUMERICAL OPERATIONS
import os                                        #FOR FILE HANDLING
import pandas as pd                              #FOR DATA HANDLING

'''
Pipeline 1: Data Extraction
In the first step, the program connects to the Alpha Vantage API, 
which is a free online service that provides stock market data. 
We tell it which stock we want (for example, Amazon symbol: AMZN) 
and ask for the daily closing prices for as many past days as possible. 
The code then downloads this information (each date and its closing
price) and stores them in a csv file in reversed order so that the data 
goes from oldest to newest (since the API sends them in reverse order). 
At the end of this step, we have a clean csv containing daily closing 
prices ready for processing.
'''

def DownloadData(Config, StockName , SavePath="Data"):

    #CREATE DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(SavePath, exist_ok=True)   
    FilePath = os.path.join(SavePath, f"{StockName}_daily.csv")

    #FETCH DATA FROM ALPHA VANTAGE API
    ts = TimeSeries(key=Config["AlphaVantage"]["API_KEY"])
    data, _ = ts.get_daily(StockName, outputsize=Config["AlphaVantage"]["Outputsize"])

    #CONVERT DATA INTO A DATAFRAME AND SAVE AS CSV
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "date"
    df.sort_index(inplace=True)
    df.to_csv(FilePath)

    print(f"Data saved locally at: {FilePath}")
    return FilePath

def GetStockFilePath(StockName, SavePath="Data"):
    #USE THE StockName TO CONSTRUCT THE FILE PATH OF THE DOWNLOADED CSV FILE FOR THIS STOCK
    return os.path.join(SavePath, f"{StockName}_daily.csv")

def FetchData(FilePath, Config):
    #LOAD DATA FROM CSV FILE
    df = pd.read_csv(FilePath)
    DataDate = df["date"].tolist()

    DataClosePrice = df[Config["AlphaVantage"]["InputFeature"]].astype(float).to_numpy()

    NumDataPoints = len(DataDate)
    DateRange = f"from {DataDate[0]} to {DataDate[-1]}"
    print("Loaded", NumDataPoints, "data points", DateRange)

    return DataDate, DataClosePrice, NumDataPoints, DateRange


def PlotRawData(DataDate, DataClosePrice, NumDataPoints, DateRange, Config):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(DataDate, DataClosePrice, color=Config["Plots"]["color_actual"])
    xticks = [DataDate[i] if ((i%Config["Plots"]["X-Interval"]==0 and (NumDataPoints-i) > Config["Plots"]["X-Interval"]) or i==NumDataPoints-1) else None for i in range(NumDataPoints)] 
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + Config["AlphaVantage"]["InputFeature"] + ", " + DateRange)
    plt.grid(True, which='major', axis='y', linestyle='--')
    plt.show()

