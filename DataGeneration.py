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

def DownloadData(Config, StockName, SavePath="Data"):
    # CREATE DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(SavePath, exist_ok=True)
    FilePath = os.path.join(SavePath, f"{StockName}_daily.csv")

    # FETCH DATA FROM ALPHA VANTAGE API
    ts = TimeSeries(key=Config["AlphaVantage"]["API_KEY"])
    data, _ = ts.get_daily(StockName, outputsize=Config["AlphaVantage"]["Outputsize"])

    # CONVERT DATA INTO A DATAFRAME
    new_df = pd.DataFrame.from_dict(data, orient="index")
    new_df.index.name = "date"
    new_df.sort_index(inplace=True)

    if os.path.exists(FilePath):
        # IF FILE EXISTS, APPEND NEW DATA
        existing_df = pd.read_csv(FilePath, index_col="date")
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        combined_df.sort_index(inplace=True)
        combined_df.to_csv(FilePath)
        print(f"Data updated and saved locally at: {FilePath}")
    else:
        # IF FILE DOES NOT EXIST, CREATE NEW FILE
        new_df.to_csv(FilePath)
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


def PlotRawData(DataDate, DataClosePrice, NumDataPoints, DateRange, Config, StockName=None):
    """
    Plot raw close price and save the figure to the PlotImages folder. If StockName is
    provided the file will be saved as {StockName}_raw.png, otherwise as raw.png.
    """
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(DataDate, DataClosePrice, color=Config["Plots"]["color_actual"])
    xticks = [DataDate[i] if ((i%Config["Plots"]["X-Interval"]==0 and (NumDataPoints-i) > Config["Plots"]["X-Interval"]) or i==NumDataPoints-1) else None for i in range(NumDataPoints)] 
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + (StockName if StockName is not None else Config["AlphaVantage"]["InputFeature"]) + ", " + DateRange)
    plt.grid(True, which='major', axis='y', linestyle='--')

    # Save the plot to PlotImages (match Plotter's save directory)
    save_dir = os.path.join(os.path.dirname(__file__), "PlotImages")
    # also include the absolute path used elsewhere for robustness
    abs_dir = r"C:\Users\sense\Desktop\SEMESTER 7\Artificial Intelligence\Project\PlotImages"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(abs_dir, exist_ok=True)

    fname = f"{StockName}_raw.png" if StockName else "raw.png"
    save_path = os.path.join(save_dir, fname)
    try:
        fig.savefig(save_path, bbox_inches="tight")
        # also save to the absolute path used by other plotters for consistency
        alt_save = os.path.join(abs_dir, fname)
        if alt_save != save_path:
            fig.savefig(alt_save, bbox_inches="tight")
        print(f"Saved raw plot: {save_path}")
    except Exception as e:
        print("Failed to save raw plot:", e)

    plt.close(fig)

