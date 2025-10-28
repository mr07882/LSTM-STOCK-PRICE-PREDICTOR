from DataGeneration import FetchData , PlotRawData , DownloadData , GetStockFilePath
from Configuration import Config
from DataPrep import Normalizer , PrepDataX , PrepDataY, SplitData
from TimeSeries import TimeSeriesDataset
from LSTM_Model import LSTMModel, TrainModel , EvaluateModel , PredictNextDay
import torch
from Plotter import Plot
from XGBoost_Model import TrainXGBModel , EvaluateXGBModel , PredictNextDayXGB
from N_BEATS_Model import TrainNBeatsModel , EvaluateNBeatsModel , PredictNextDayNBeats


#TASK 1: MEASURE THE BEST PARAMETERS FOR THE LSTM MODEL BY TRAIL AND ERROR

'''
WE WILL VARRY EACH PARAMETER AND SEE HOW IT AFFECTS THE MODEL'S PERFORMANCE/ACCURACY.
PARAMETERS TO VARY:
- NUMBER OF LAYERS
- HIDDEN LAYER SIZE
- DROPOUT RATE
- LEARNING RATE
- BATCH SIZE
- NUMBER OF EPOCHS
- OPTIMIZER TYPE
- LOSS FUNCTION
THESE EXPERIMENTS WILL HELP US UNDERSTAND THE IMPACT OF EACH PARAMETER. THESE WILL BE DISPLAYED IN A PLOT
OF ACCURACY AGAINST PARAMETER VALUE FOR EACH PARAMETER.
'''


def run_lstm_param_sweep(stock_symbol: str,
						 param_name: str,
						 values: list,
						 quick: bool = True):
	"""Run a simple parameter sweep for the LSTM model varying one parameter at a time.

	This function will:
	- load the stock csv using GetStockFilePath/FetchData
	- prepare data (normalize, window, split)
	- for each value: set the config for that parameter, train a short model, evaluate and collect accuracy
	- plot accuracy vs parameter value

	quick=True uses fewer epochs (Config Training EPOCHS is temporarily reduced) so the sweep runs faster.
	"""
	print(f"Running LSTM parameter sweep for {param_name} on {stock_symbol}...")

	# fetch data
	filepath = GetStockFilePath(stock_symbol)
	DataDate, DataClosePrice, NumDataPoints, DateRange = FetchData(filepath, Config)

	# prepare data
	Scalar = Normalizer()
	NormalizedDataClosePrice = Scalar.FitTransformation(DataClosePrice)
	data_x, data_x_unseen = PrepDataX(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"]) 
	data_y = PrepDataY(NormalizedDataClosePrice, PredictionCycle=Config["Data"]["PredictionCycle"]) 

	SplitIndex = int(data_y.shape[0]*Config["Data"]["DataSplitRatio"]) 
	TrainingData_Input, TestingData_Input, TrainingData_Output, TestingData_Output = SplitData(data_x, data_y, SplitIndex)

	TrainingDataset = TimeSeriesDataset(TrainingData_Input, TrainingData_Output)
	TestingDataset = TimeSeriesDataset(TestingData_Input, TestingData_Output)

	# Save original values so we can restore config
	original_training = Config["Training"].copy()
	original_model = Config["Model"].copy()

	# If quick, reduce epochs to speed up
	if quick:
		Config["Training"]["EPOCHS"] = max(2, min(10, Config["Training"]["EPOCHS"]))

	results = []

	for v in values:
		print(f"Testing {param_name} = {v}")

		# apply parameter to Config or to local model constructor arguments
		# support common param names
		if param_name.lower() in ("num_layers", "numoflstmayers", "numoflstmayers", "num_of_layers"):
			Config["Model"]["NumOfLSTMLayers"] = int(v)
		elif param_name.lower() in ("hidden_layer_size", "lstm_size", "hidden_size"):
			Config["Model"]["LSTMSize"] = int(v)
		elif param_name.lower() in ("dropout",):
			Config["Model"]["Dropout"] = float(v)
		elif param_name.lower() in ("learning_rate", "lr"):
			Config["Training"]["LearningRate"] = float(v)
		elif param_name.lower() in ("batch_size", "batch"):
			Config["Training"]["BatchSize"] = int(v)
		elif param_name.lower() in ("epochs", "num_epochs"):
			Config["Training"]["EPOCHS"] = int(v)
		else:
			print(f"Parameter {param_name} not recognized, skipping value {v}")
			continue

		# build model using current config
		model = LSTMModel(input_size=Config["Model"]["InputSize"],
						  hidden_layer_size=Config["Model"]["LSTMSize"],
						  num_layers=Config["Model"]["NumOfLSTMLayers"],
						  output_size=1,
						  dropout=Config["Model"]["Dropout"]) 

		# train and evaluate
		model = TrainModel(model, TrainingDataset, TestingDataset, Config)
		train_preds, test_preds, mape, accuracy = EvaluateModel(model, TrainingDataset, TestingDataset, Scalar, Config)

		results.append((v, float(accuracy)))

	# restore original config
	Config["Training"] = original_training
	Config["Model"] = original_model

	# plot results
	vals = [r[0] for r in results]
	acc = [r[1] for r in results]

	import matplotlib.pyplot as plt
	plt.figure(figsize=(8,4))
	plt.plot(vals, acc, marker='o')
	plt.title(f"LSTM accuracy vs {param_name} ({stock_symbol})")
	plt.xlabel(param_name)
	plt.ylabel('Accuracy (%)')
	plt.grid(True)
	plt.show()


if __name__ == '__main__':
	# Example quick sweeps (small and fast). Set quick=False to use full epochs from Config.
	# Run a quick sweep over hidden layer sizes
	run_lstm_param_sweep(stock_symbol=Config["AlphaVantage"]["Stock"], param_name='hidden_layer_size', values=[8,16,32,64], quick=True)
	# Run a quick sweep over number of LSTM layers
	run_lstm_param_sweep(stock_symbol=Config["AlphaVantage"]["Stock"], param_name='num_layers', values=[1,2,3], quick=True)










#TASK 2: MEASURE THE BEST PARAMETERS FOR THE XGBOOST MODEL BY TRAIL AND ERROR

'''
WE WILL VARY EACH PARAMETER AND SEE HOW IT AFFECTS THE MODEL'S PERFORMANCE/ACCURACY.
PARAMETERS TO VARY:
- MAX DEPTH
- SUBSAMPLE RATIO
- COLUMN SAMPLE BY TREE
- REGULARIZATION PARAMETER (LAMBDA)
- RANDOM STATE
THESE EXPERIMENTS WILL HELP US UNDERSTAND THE IMPACT OF EACH PARAMETER. THESE WILL BE DISPLAYED IN A PLOT
OF ACCURACY AGAINST PARAMETER VALUE FOR EACH PARAMETER.
'''









#TASK 3: MEASURE THE BEST PARAMETERS FOR THE N-BEATS MODEL BY TRAIL AND ERROR

'''
WE WILL VARY EACH PARAMETER AND SEE HOW IT AFFECTS THE MODEL'S PERFORMANCE/ACCURACY.
PARAMETERS TO VARY:
- NUMBER OF STACKS
- NUMBER OF BLOCKS
- HIDDEN LAYER SIZE
- DROPOUT RATE
THESE EXPERIMENTS WILL HELP US UNDERSTAND THE IMPACT OF EACH PARAMETER. THESE WILL BE DISPLAYED IN A PLOT
OF ACCURACY AGAINST PARAMETER VALUE FOR EACH PARAMETER.
'''









#TASK 4: SELECTING THE BEST MODEL

'''
WE WILL RUN EACH MODEL ON ALL THE 10 STOCKS AND COMPARE THEIR AVERAGE ACCURACY.
THE MODEL WITH THE HIGHEST AVERAGE ACCURACY WILL BE SELECTED AS THE BEST MODEL FOR STOCK PRICE PREDICTION.
THIS WILL HELP US UNDERSTAND WHICH MODEL IS MOST EFFECTIVE FOR THIS TASK.
'''











#TASK 5: MODEL PREFERENCES BASED ON STOCK CHARACTERISTICS

'''
WE WILL ANALYZE THE PERFORMANCE OF EACH MODEL BASED ON DIFFERENT STOCK CHARACTERISTICS
SUCH AS VOLATILITY, TRENDINESS, AND SEASONALITY. WE WILL GROUP STOCKS BASED ON THESE
CHARACTERISTICS AND EVALUATE WHICH MODEL PERFORMS BEST FOR EACH GROUP.
THIS WILL HELP US UNDERSTAND IF CERTAIN MODELS ARE BETTER SUITED FOR SPECIFIC TYPES OF STOCKS.
'''