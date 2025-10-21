
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def Plot(
    DataDate,
    TrainingData_Output,
    TestingData_Output,
    Scalar,
    NumDataPoints,
    SplitIndex,
    Config,
    Stock,
    mode="split",
    DataClosePrice=None,
    TrainingPredictions=None,
    TestingPredictions=None,
    Prediction=None
):
    """
    Multi-mode plotter for stock price training, validation, and prediction comparison.

    Modes
    -----
    "split"     : Plots training vs validation actual data (original)
    "predicted" : Plots actual vs predicted prices (full)
    "nextday"   : Plots last few days + predicted next-day price
    """

    print(f"Preparing data for plotting ({mode})...")
    cycle = Config["Data"]["PredictionCycle"]

    # ---------------------------------------------------------------
    # MODE 1: TRAIN/VALIDATION SPLIT
    # ---------------------------------------------------------------
    if mode == "split":
        to_plot_TrainingData_Output = np.zeros(NumDataPoints)
        to_plot_TestingData_Output = np.zeros(NumDataPoints)

        to_plot_TrainingData_Output[cycle:SplitIndex + cycle] = Scalar.InverseTransformation(TrainingData_Output)
        to_plot_TestingData_Output[SplitIndex + cycle:] = Scalar.InverseTransformation(TestingData_Output)

        to_plot_TrainingData_Output = np.where(to_plot_TrainingData_Output == 0, None, to_plot_TrainingData_Output)
        to_plot_TestingData_Output = np.where(to_plot_TestingData_Output == 0, None, to_plot_TestingData_Output)

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
        plt.xticks(np.arange(0, len(xticks)), xticks, rotation='vertical')
        plt.title("Daily close prices for " + Stock + " - showing training and validation data")
        plt.grid(True, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()
        return None

    # ---------------------------------------------------------------
    # MODE 2: FULL PREDICTED VS ACTUAL
    # ---------------------------------------------------------------
    elif mode == "predicted":
        if DataClosePrice is None or TrainingPredictions is None or TestingPredictions is None:
            raise ValueError("For mode='predicted', please pass DataClosePrice, TrainingPredictions, and TestingPredictions.")

        to_plot_data_y_train_pred = np.zeros(NumDataPoints)
        to_plot_data_y_val_pred = np.zeros(NumDataPoints)

        to_plot_data_y_train_pred[cycle:SplitIndex + cycle] = Scalar.InverseTransformation(TrainingPredictions)
        to_plot_data_y_val_pred[SplitIndex + cycle:] = Scalar.InverseTransformation(TestingPredictions)

        to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

        print("Plotting predicted vs actual prices...")
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(DataDate, DataClosePrice, label="Actual prices", color=Config["Plots"]["color_actual"])
        plt.plot(DataDate, to_plot_data_y_train_pred, label="Predicted prices (train)", color=Config["Plots"]["color_pred_train"])
        plt.plot(DataDate, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=Config["Plots"]["color_pred_val"])

        xticks = [
            DataDate[i] if (
                (i % Config["Plots"]["X-Interval"] == 0 and (NumDataPoints - i) > Config["Plots"]["X-Interval"])
                or i == NumDataPoints - 1
            ) else None for i in range(NumDataPoints)
        ]
        plt.xticks(np.arange(0, len(xticks)), xticks, rotation='vertical')
        plt.title("Compare predicted prices to actual prices")
        plt.grid(True, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

        return None

    # ---------------------------------------------------------------
    # MODE 3: NEXT-DAY PREDICTION VISUALIZATION
    # ---------------------------------------------------------------
    elif mode == "nextday":
        if Prediction is None or TestingPredictions is None or TestingData_Output is None:
            raise ValueError("For mode='nextday', please pass Prediction, TestingPredictions, and TestingData_Output.")

        print("Plotting last few days + next-day prediction...")
        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        # Last few days
        to_plot_data_y_val[:plot_range-1] = Scalar.InverseTransformation(TestingData_Output)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = Scalar.InverseTransformation(TestingPredictions)[-plot_range+1:]
        # Tomorrow
        to_plot_data_y_test_pred[plot_range-1] = Scalar.InverseTransformation(Prediction)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        plot_date_test = DataDate[-plot_range+1:]
        plot_date_test.append("tomorrow")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=Config["Plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=Config["Plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=Config["Plots"]["color_pred_test"])
        plt.title("Predicting the close price of the next trading day")
        plt.grid(True, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

        NextDayPrice = to_plot_data_y_test_pred[plot_range - 1]
        print(f"Predicted close price of the next trading day: {round(NextDayPrice, 2)}")
        return NextDayPrice
        

    else:
        raise ValueError("Invalid mode. Choose 'split', 'predicted', or 'nextday'.")


