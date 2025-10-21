import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor


def TrainXGBModel(TrainingData_Input, TrainingData_Output, TestingData_Input, TestingData_Output):
    Model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42
    )
    print("PROCESS: Training XGBoost model................")
    Model.fit(TrainingData_Input, TrainingData_Output, eval_set=[(TestingData_Input, TestingData_Output)], verbose=False)
    print("CHECKPOINT 1: Training Completed.")
    return Model


def EvaluateXGBModel(model, TrainingData_Input, TestingData_Input, TestingData_Output, Scalar):
    print("PROCESS: Evaluating XGBoost model on testing data................")

    #PREDICTIONS MADE BY THE MODEL
    TrainingPredictions = model.predict(TrainingData_Input)
    TestingPredictions = model.predict(TestingData_Input)

    # USING INVERSE TRANSFORMATIONS TO CONVERT OUTPUT INTO PRICE
    TruePrices = Scalar.InverseTransformation(TestingData_Output)
    PredictedPrices = Scalar.InverseTransformation(TestingPredictions)

    # EVALUATING ON THE BASIS OF MAPE AND ACCURACY
    MAPE = mean_absolute_percentage_error(TruePrices, PredictedPrices) * 100
    Accuracy = 100 - MAPE

    print(f"\nRegression Accuracy (based on MAPE): {Accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE:.2f}%")

    print("CHECKPOINT 2: Evaluation Completed.")
    return TrainingPredictions, TestingPredictions, MAPE, Accuracy


def PredictNextDayXGB(model, LastWindow):
    """
    Predicts the next day's closing price based on the latest input window.
    LastWindow should be a 1D array of length window_size.
    """
    print("PROCESS: Predicting next day's closing price using XGBoost................")
    arr = np.array(LastWindow)
    if arr.ndim == 2:
        arr = arr[-1]
    LastWindow = arr.reshape(1, -1)  # ensure 2D
    Prediction = model.predict(LastWindow)

    # Inverse transform to price scale
    # try:
    #     NextDayPrice = Scalar.InverseTransformation(Prediction)
    # except Exception:
    #     NextDayPrice = Scalar.inverse_transform(Prediction)

    # print(f"Predicted next-day closing price: {NextDayPrice[0]:.2f}")

    # return float(NextDayPrice[0])
    print("CHECKPOINT 3: Prediction Completed.")
    return Prediction


