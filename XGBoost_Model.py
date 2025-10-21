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


def EvaluateXGBModel(model, TestingData_Input, TestingData_Output, Scalar):
    """
    Evaluates the trained model on validation data.
    Returns MAPE and Accuracy.
    """
    print("Evaluating model on validation data...")
    TestingPredictions = model.predict(TestingData_Input)

    # Inverse transform to actual prices
    TruePrices = Scalar.InverseTransformation(TestingData_Output)
    PredictedPrices = Scalar.InverseTransformation(TestingPredictions)

    # Calculate MAPE and Accuracy
    MAPE = mean_absolute_percentage_error(TruePrices, PredictedPrices) * 100
    Accuracy = 100 - MAPE

    print(f"\nRegression Accuracy (based on MAPE): {Accuracy:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE:.2f}%")

    return MAPE, Accuracy, TestingPredictions


def PredictNextDayXGB(model, LastWindow, Scalar):
    """
    Predicts the next day's closing price based on the latest input window.
    """
    print("Predicting next day's closing price...")
    LastWindow = LastWindow.reshape(1, -1)  # ensure 2D
    Prediction = model.predict(LastWindow)

    # Inverse transform to price scale
    NextDayPrice = Scalar.InverseTransformation(Prediction)
    print(f"Predicted next-day closing price: {NextDayPrice[0]:.2f}")

    return float(NextDayPrice[0])


