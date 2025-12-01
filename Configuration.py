Config = {
    "AlphaVantage": {
        "API_KEY": "M1QSTF80I3S3AFXV",       # https://www.alphavantage.co/support/#api-key
        "Stock": "NULL",                     #NVDA , AMZN , MSFT #TARGET STOCK
        "Outputsize": "compact",                #FETCH ENITRE TIME SERIES DATA IE FROM TIME AVAILABLE TILL TODAY 
        "InputFeature": "4. close",          #FEATURE TO BE USED FROM TIME SERIES DATA: CLOSING STOCK PRICE 
    },
    "Data": {  
        "PredictionCycle": 60,               #NUMBER OF PREVIOUS DAYS TO CONSIDER FOR PREDICTION
        "DataSplitRatio": 0.80,              #TRAIN-TEST SPLIT RATIO
    },
    "Plots": {
        "X-Interval": 90,                    #INTERVAL FOR X-AXIS IN PLOTS   
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "Model": {
        "InputSize": 1,                     #NUMBER OF INPUT FEATURES: IN THIS CASE, ONLY THE CLOSING PRICE
        "NumOfLSTMLayers": 2,               #NUMBER OF LSTM LAYERS
        "LSTMSize": 32,                     #NUMBER OF NEURONS IN EACH LSTM LAYER
        "Dropout": 0.2,                     #DROPOUT RATE FOR REGULARIZATION
    },
    "Training": {
        "Device": "cpu",                    #DEVICE TO RUN THE MODEL ON: GPU (CUDA) OR CPU
        "BatchSize": 64,                    #BATCH SIZE FOR TRAINING
        "EPOCHS": 100,                      #NUMBER OF EPOCHS FOR TRAINING
        "LearningRate": 0.01,               #INITIAL LEARNING RATE FOR OPTIMIZER
        "StepSize": 40,                     #STEP SIZE FOR LEARNING RATE SCHEDULER
    }
}