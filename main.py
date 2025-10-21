from DataGeneration import FetchData , PlotRawData , DownloadData , GetStockFilePath
from Configuration import Config
from DataPrep import Normalizer , PrepDataX , PrepDataY, SplitData
from TimeSeries import TimeSeriesDataset
from LSTM_Model import LSTMModel, TrainModel , EvaluateModel , PredictNextDay
import torch
from Plotter import Plot
from XGBoost_Model import TrainXGBModel , EvaluateXGBModel , PredictNextDayXGB


while True:
        print("------------------MENU---------------------")
        print("1. Download Historical Stock Data")
        print("2. Predict Stock Closing Price")
        print("0. Exit")

        Choice = input("Enter your choice: ").strip()
        if Choice == "0":
            print("Exiting the program. Goodbye!")
            break

        if Choice == "1":
            print("Available Stocks: AMZN, MSFT, NVDA, GOOG, META, INTC, TSLA, WMT, NKE, MCD")
            StockName = input("Enter the stock name: ").strip().upper()
            print(f"Downloading historical data for {StockName}...")

            #-------------PIPELINE 1--------------
            FilePath = DownloadData(Config , StockName)
            break

        elif Choice == "2":
            
            print("Available Stocks: AMZN, MSFT, NVDA, GOOG, META, INTC, TSLA, WMT, NKE, MCD")
            StockName = input("Enter the stock name: ").strip().upper()
            print(f"Predicting closing price for {StockName}...")

            FilePath = GetStockFilePath(StockName)
            print(f"FETCHING HISTORICAL DATA FOR {StockName}...")
            DataDate, DataClosePrice, NumDataPoints, DateRange = FetchData(FilePath, Config)
            print(f"PLOTTING HISTORICAL DATA FOR {StockName}...")
            PlotRawData(DataDate, DataClosePrice, NumDataPoints, DateRange, Config)
            
            #-------------PIPELINE 2--------------
            print("PREPARING DATA FOR TRAINING...")

            #STEP 1: NORMALIZE DATA BETWEEN [-1, 1]
            Scalar = Normalizer()
            NormalizedDataClosePrice = Scalar.FitTransformation(DataClosePrice)
            data_x, data_x_unseen = PrepDataX(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"])
            data_y = PrepDataY(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"])

            #STEP 2: SPLITTING DATA INTO TRAINING AND TESTING DATA 
            SplitIndex = int(data_y.shape[0]*Config["Data"]["DataSplitRatio"])
            TrainingData_Input, TestingData_Input, TrainingData_Output, TestingData_Output = SplitData(data_x, data_y, SplitIndex)
            Temp = Plot(DataDate, TrainingData_Output, TestingData_Output, Scalar, NumDataPoints, SplitIndex, Config ,StockName, mode="split")
    
            #STEP 3:CONVERTING EACH DATA INTO TORCH DATASET 
            TrainingDataset = TimeSeriesDataset(TrainingData_Input, TrainingData_Output)
            TestingDataset = TimeSeriesDataset(TestingData_Input, TestingData_Output)

            print("DATA IS READY! ")
            print("Models Available: LSTM, XGBOOST")
            ModelName = input("Enter The Model Name: ").strip().upper()

            if ModelName == "LSTM":
                #-------------PIPELINE 3--------------
                #BUILDING THE LSTM MODEL
                Model = LSTMModel(input_size=Config["Model"]["InputSize"],
                                hidden_layer_size=Config["Model"]["LSTMSize"],
                                num_layers=Config["Model"]["NumOfLSTMLayers"],
                                output_size=1,
                                dropout=Config["Model"]["Dropout"]) 

                #-------------PIPELINE 4--------------

                #STEP 1: DECIDING WHETHER CPU OR GPU SHOULD BE USED FOR TRAINING THE MODEL
                if isinstance(Config["Training"]["Device"], str) and Config["Training"]["Device"] == "cpu":
                    torch_device = torch.device('cpu')
                else:
                    torch_device = torch.device(Config["Training"]["Device"]) if isinstance(Config["Training"]["Device"], str) else Config["Training"]["Device"]
    
                #STEP 2: TRAIN AND EVALUATE THE MODEL
                Model = TrainModel(Model, TrainingDataset, TestingDataset, Config)

                #-------------PIPELINE 5--------------
                #STEP 1: EVALUATE THE MODEL ON TRAINING AND TESTING DATA
                TrainingPredictions, TestingPredictions, mape , accuracy = EvaluateModel(Model, TrainingDataset, TestingDataset, Scalar,Config)
                                                                                                
                Temp = Plot(
                        DataDate=DataDate,
                        TrainingData_Output=None,
                        TestingData_Output=None,
                        Scalar=Scalar,
                        NumDataPoints=NumDataPoints,
                        SplitIndex=SplitIndex,
                        Config=Config,
                        Stock=StockName,
                        mode="predicted",
                        DataClosePrice=DataClosePrice,
                        TrainingPredictions=TrainingPredictions,
                        TestingPredictions=TestingPredictions
                    )
                
                #STEP 2: PREDICT THE CLOSE PRICE FOR THE NEXT DAY
                Prediction = PredictNextDay(Model, TestingData_Input,Config)
                                            
                NextDayPrice = Plot(
                    DataDate=DataDate,
                    TrainingData_Output=None,
                    TestingData_Output=TestingData_Output,
                    Scalar=Scalar,
                    NumDataPoints=NumDataPoints,
                    SplitIndex=SplitIndex,
                    Config=Config,
                    Stock=StockName,
                    mode="nextday",
                    TestingPredictions=TestingPredictions,
                    Prediction=Prediction
                )
                break
            elif ModelName == "XGBOOST":
                #-------------PIPELINE 4--------------
                model = TrainXGBModel(TrainingData_Input, TrainingData_Output, TestingData_Input, TestingData_Output)
                
                #-------------PIPELINE 5--------------
                TrainingPredictions, TestingPredictions, mape, accuracy = EvaluateXGBModel(
                    model,
                    TrainingData_Input,
                    TrainingData_Output,
                    TestingData_Input,
                    TestingData_Output,
                    Scalar,
                )

                Temp = Plot(
                        DataDate=DataDate,
                        TrainingData_Output=None,
                        TestingData_Output=None,
                        Scalar=Scalar,
                        NumDataPoints=NumDataPoints,
                        SplitIndex=SplitIndex,
                        Config=Config,
                        Stock=StockName,
                        mode="predicted",
                        DataClosePrice=DataClosePrice,
                        TrainingPredictions=TrainingPredictions,
                        TestingPredictions=TestingPredictions
                    )
                
                #NextDayPrice = PredictNextDayXGB(model, TestingData_Input[-1], Scalar)
                Prediction = PredictNextDayXGB(model, TestingData_Input[-1], Scalar)
                NextDayPrice = Plot(
                    DataDate=DataDate,
                    TrainingData_Output=None,
                    TestingData_Output=TestingData_Output,
                    Scalar=Scalar,
                    NumDataPoints=NumDataPoints,
                    SplitIndex=SplitIndex,
                    Config=Config,
                    Stock=StockName,
                    mode="nextday",
                    TestingPredictions=TestingPredictions,
                    Prediction=Prediction
                )
                break

        else:
            print("Invalid input, please try again.\n")


