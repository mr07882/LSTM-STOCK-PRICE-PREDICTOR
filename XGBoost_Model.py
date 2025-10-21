import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor


def TrainXGBModel(TrainingData_Input, TrainingData_Output, TestingData_Input, TestingData_Output):
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42
    )

    print("Training XGBoost model...")
    model.fit(TrainingData_Input, TrainingData_Output, eval_set=[(TestingData_Input, TestingData_Output)], verbose=False)
    print("Training complete.")
    return model


def EvaluateXGBModel(model, TrainingData_Input, TrainingData_Output, TestingData_Input, TestingData_Output, Scalar):
    """
    Evaluates the trained XGBoost model on both training and validation data.

    Returns
    -------
    TrainingPredictions, TestingPredictions, MAPE, Accuracy
    """
    print("Evaluating model on training and validation data...")

    # Predictions
    TrainingPredictions = model.predict(TrainingData_Input)
    TestingPredictions = model.predict(TestingData_Input)

    # Inverse transform to actual prices (use project's scaler API)
    try:
        true_prices = Scalar.InverseTransformation(TestingData_Output)
        pred_prices = Scalar.InverseTransformation(TestingPredictions)
    except Exception:
        # fallback to common sklearn-style scaler if method names differ
        true_prices = Scalar.inverse_transform(TestingData_Output)
        pred_prices = Scalar.inverse_transform(TestingPredictions)

    # Calculate MAPE and Accuracy
    MAPE = mean_absolute_percentage_error(true_prices, pred_prices) * 100
    Accuracy = 100 - MAPE

    print(f"\nRegression Accuracy (based on MAPE): {Accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE:.2f}%")

    return TrainingPredictions, TestingPredictions, MAPE, Accuracy


def PredictNextDayXGB(model, LastWindow, Scalar):
    """
    Predicts the next day's closing price based on the latest input window.
    LastWindow should be a 1D array of length window_size.
    """
    print("Predicting next day's closing price...")
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
    return Prediction


