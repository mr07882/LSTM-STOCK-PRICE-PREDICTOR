# ===========================================================
#     STOCK PRICE PREDICTION USING XGBOOST + NORMALIZATION
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

print("All libraries loaded.")

# ---------------- CONFIG ----------------
config = {
    "alpha_vantage": {
        "key": "N2D1KYB158XO1ZAK",
        "symbol": "AMZN",
        "outputsize": "full",
        "key_adjusted_close": "4. close",
    },
    "data": {
        "window_size": 60,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_val": "#FF851B",
    }
}

# ---------------- DATA EXTRACTION ----------------
def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, _ = ts.get_daily(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    data_date = list(data.keys())
    data_date.reverse()
    data_close = [float(data[d][config["alpha_vantage"]["key_adjusted_close"]]) for d in data.keys()]
    data_close.reverse()
    return data_date, np.array(data_close)

data_date, data_close_price = download_data(config)
print(f"Fetched {len(data_close_price)} points for {config['alpha_vantage']['symbol']}")

# ---------------- NORMALIZATION ----------------
class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None
    def fit_transform(self, x):
        self.mu = np.mean(x)
        self.sd = np.std(x)
        return (x - self.mu) / self.sd
    def inverse_transform(self, x):
        return (x * self.sd) + self.mu

scaler = Normalizer()
normalized_prices = scaler.fit_transform(data_close_price)

# ---------------- WINDOWING ----------------
window = config["data"]["window_size"]
X, y = [], []
for i in range(window, len(normalized_prices)):
    X.append(normalized_prices[i-window:i])
    y.append(normalized_prices[i])
X, y = np.array(X), np.array(y)

split_idx = int(len(y) * config["data"]["train_split_size"])
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# ---------------- TRAIN XGBOOST ----------------
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ---------------- PREDICTIONS ----------------
y_pred_val = model.predict(X_val)

# Inverse transform predictions to actual price scale
true_prices = scaler.inverse_transform(y_val)
pred_prices = scaler.inverse_transform(y_pred_val)

# ---------------- ACCURACY ----------------
mape = mean_absolute_percentage_error(true_prices, pred_prices) * 100
accuracy = 100 - mape
print(f"\nRegression Accuracy (based on MAPE): {accuracy:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# ---------------- PLOTS ----------------
num_points = len(data_close_price)
val_start = window + split_idx
val_dates = data_date[val_start:val_start+len(y_val)]

fig = figure(figsize=(25,5), dpi=80)
plt.plot(data_date, data_close_price, label="Actual Prices", color=config["plots"]["color_actual"])
plt.plot(val_dates, pred_prices, label="Predicted Prices (Validation)", color=config["plots"]["color_pred_val"])
plt.title(f"XGBoost Predicted vs Actual Closing Prices for {config['alpha_vantage']['symbol']}")
plt.xticks(rotation='vertical')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

# ---------------- NEXT-DAY FORECAST ----------------
latest_window = normalized_prices[-window:]
next_norm = model.predict(latest_window.reshape(1, -1))[0]
next_price = scaler.inverse_transform(next_norm)
print(f"Predicted next-day closing price: {next_price:.2f}")
