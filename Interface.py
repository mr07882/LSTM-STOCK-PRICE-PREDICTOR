import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import atexit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DataGeneration import FetchData, PlotRawData, DownloadData, GetStockFilePath
from Configuration import Config
from DataPrep import Normalizer, PrepDataX, PrepDataY, SplitData
from TimeSeries import TimeSeriesDataset
from LSTM_Model import LSTMModel, TrainModel, EvaluateModel, PredictNextDay
from XGBoost_Model import TrainXGBModel, EvaluateXGBModel, PredictNextDayXGB
from N_BEATS_Model import TrainNBeatsModel, EvaluateNBeatsModel, PredictNextDayNBeats
from Plotter import Plot


STOCKS = ["AAPL", "ADBE", "AMZN", "ASML", "GOOG", "IBM", "INTC", "MCD", "META" , "MSFT", "NFLX" , "NKE" , "NVDA" , "ORCL" , "SONY" , "TSLA" , "WMT" ]
MODELS = ["LSTM", "XGBOOST", "N-BEATS"]

# Color scheme
COLORS = {
    'bg_primary': '#1a1d29',
    'bg_secondary': '#252936',
    'bg_tertiary': '#2d3142',
    'accent_blue': '#4a90e2',
    'accent_green': '#50c878',
    'accent_purple': '#9b59b6',
    'accent_orange': '#ff6b35',
    'text_primary': '#ffffff',
    'text_secondary': '#b0b8c4',
    'border': '#3a3f51',
    'hover': '#364156'
}


class StockPredictorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Price Predictor - AI Analysis Platform")
        self.geometry("1200x750")
        self.configure(bg=COLORS['bg_primary'])
        
        # Register cleanup on exit
        atexit.register(self._cleanup_saved_plots)
        
        # Apply custom style
        self._setup_styles()
        
        # State
        self.selected_stock = tk.StringVar(value=STOCKS[0])
        self.selected_model = tk.StringVar(value=MODELS[0])
        self.scalar = None
        self.data_date = None
        self.data_close = None
        self.num_points = None
        self.split_index = None
        self.training_preds = None
        self.testing_preds = None
        self.prediction_next = None
        # Results display variables
        self.result_nextday_var = tk.StringVar(value="N/A")
        self.result_acc_var = tk.StringVar(value="N/A")
        self.result_mape_var = tk.StringVar(value="N/A")
        
        self._build_ui()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=COLORS['bg_primary'])
        style.configure('Card.TFrame', background=COLORS['bg_secondary'], relief='flat')
        style.configure('TLabel', background=COLORS['bg_primary'], foreground=COLORS['text_primary'], 
                       font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=COLORS['bg_primary'], foreground=COLORS['text_primary'],
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', background=COLORS['bg_tertiary'], foreground=COLORS['accent_green'],
                       font=('Segoe UI', 9), padding=8, relief='flat')
        
        # Buttons
        style.configure('Action.TButton', background=COLORS['accent_blue'], foreground=COLORS['text_primary'],
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none', padding=10)
        style.map('Action.TButton',
                 background=[('active', COLORS['hover']), ('pressed', COLORS['accent_purple'])])
        
        # Radiobuttons
        style.configure('Stock.TRadiobutton', background=COLORS['bg_secondary'], 
                       foreground=COLORS['text_secondary'], font=('Segoe UI', 9),
                       indicatorcolor=COLORS['accent_blue'], borderwidth=0)
        style.map('Stock.TRadiobutton',
                 background=[('active', COLORS['hover'])],
                 foreground=[('selected', COLORS['accent_blue'])])
        
        # Combobox
        style.configure('TCombobox', fieldbackground=COLORS['bg_tertiary'], 
                       background=COLORS['bg_tertiary'], foreground=COLORS['text_primary'],
                       arrowcolor=COLORS['accent_blue'], borderwidth=1, relief='flat')
        
        # Notebook
        style.configure('TNotebook', background=COLORS['bg_primary'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['bg_tertiary'], 
                       foreground=COLORS['text_secondary'], padding=[20, 10],
                       borderwidth=0, font=('Segoe UI', 9))
        style.map('TNotebook.Tab',
                 background=[('selected', COLORS['accent_blue'])],
                 foreground=[('selected', COLORS['text_primary'])],
                 expand=[('selected', [1, 1, 1, 0])])
    
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS['bg_secondary'], height=70)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="📈 Stock Price Predictor", 
                              bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                              font=('Segoe UI', 18, 'bold'))
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        subtitle = tk.Label(header, text="AI-Powered Market Analysis", 
                           bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'],
                           font=('Segoe UI', 10))
        subtitle.pack(side=tk.LEFT, padx=(0, 20))
        
        # Refresh button to clear saved plot images and UI plots
        btn_refresh = self._create_action_button(header, "🔄 Refresh", self._refresh, COLORS['accent_purple'])
        btn_refresh.pack(side=tk.RIGHT, padx=20, pady=16)
        
        # Status area moved to the top-right near the refresh button
        status_frame = tk.Frame(header, bg=COLORS['bg_tertiary'], relief='flat', bd=0)
        status_frame.pack(side=tk.RIGHT, padx=(0, 20), pady=16)

        status_label = tk.Label(status_frame, text="STATUS",
                               bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'],
                               font=('Segoe UI', 9, 'bold'))
        status_label.pack(anchor=tk.W, padx=5, pady=(0, 5))

        self.status = tk.Label(status_frame, text="Ready",
                              bg=COLORS['bg_tertiary'], fg=COLORS['accent_green'],
                              font=('Segoe UI', 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Main container (scrollable)
        body = tk.Frame(self, bg=COLORS['bg_primary'])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=15)

        canvas = tk.Canvas(body, bg=COLORS['bg_primary'], highlightthickness=0)
        v_scroll = ttk.Scrollbar(body, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # This frame will hold the actual UI content and live inside the canvas
        main = tk.Frame(canvas, bg=COLORS['bg_primary'])
        canvas.create_window((0, 0), window=main, anchor='nw')

        # Keep the scrollregion updated
        def _on_frame_config(event):
            try:
                canvas.configure(scrollregion=canvas.bbox('all'))
            except Exception:
                pass

        main.bind('<Configure>', _on_frame_config)

        # Mouse wheel scrolling (Windows)
        def _on_mousewheel(event):
            # event.delta is multiple of 120 on Windows
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            except Exception:
                pass

        # Bind to the canvas so scrolling works when pointer is over the UI
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        # Left panel (controls)
        left = tk.Frame(main, bg=COLORS['bg_secondary'], width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)
        
        # Add padding frame
        left_inner = tk.Frame(left, bg=COLORS['bg_secondary'])
        left_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Stock Selection Section
        stock_frame = tk.Frame(left_inner, bg=COLORS['bg_tertiary'], relief='flat', bd=0)
        stock_frame.pack(fill=tk.X, pady=(0, 15))
        
        stock_header = tk.Label(stock_frame, text="SELECT STOCK", 
                               bg=COLORS['bg_tertiary'], fg=COLORS['accent_blue'],
                               font=('Segoe UI', 10, 'bold'))
        stock_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        # Stock grid with styled buttons
        grid = tk.Frame(stock_frame, bg=COLORS['bg_tertiary'])
        grid.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        for i, s in enumerate(STOCKS):
            b = ttk.Radiobutton(grid, text=s, value=s, variable=self.selected_stock,
                               style='Stock.TRadiobutton')
            b.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=5)
        
        # Model Selection Section
        model_frame = tk.Frame(left_inner, bg=COLORS['bg_tertiary'], relief='flat', bd=0)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        model_header = tk.Label(model_frame, text="SELECT MODEL", 
                               bg=COLORS['bg_tertiary'], fg=COLORS['accent_purple'],
                               font=('Segoe UI', 10, 'bold'))
        model_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        model_combo = ttk.Combobox(model_frame, values=MODELS, state="readonly", 
                                  textvariable=self.selected_model)
        model_combo.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Action buttons with icons
        btn_download = self._create_action_button(left_inner, "⬇ Download Data", 
                                                   self._download_data, COLORS['accent_green'])
        btn_download.pack(fill=tk.X, pady=(0, 10))
        
        btn_prepare = self._create_action_button(left_inner, "⚙ Load & Prepare", 
                                                  self._load_and_prepare, COLORS['accent_blue'])
        btn_prepare.pack(fill=tk.X, pady=(0, 10))
        
        btn_train = self._create_action_button(left_inner, "🚀 Train & Predict", 
                                                self._train_and_predict, COLORS['accent_orange'] )
        btn_train.pack(fill=tk.X, pady=(0, 10))
        
        # Spacer
        tk.Frame(left_inner, bg=COLORS['bg_secondary'], height=20).pack()
        
        # Right panel (visualization)
        right = tk.Frame(main, bg=COLORS['bg_primary'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Notebook with tabs
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(expand=True, fill=tk.BOTH, pady=(0, 10))
        
        # Create tabs with styled plots
        self._create_plot_tab("📊 Raw Data", 'fig_raw', 'ax_raw', 'canvas_raw')
        self._create_plot_tab("✂️ Train/Test Split", 'fig_split', 'ax_split', 'canvas_split')
        self._create_plot_tab("🎯 Predictions", 'fig_pred', 'ax_pred', 'canvas_pred')
        self._create_plot_tab("🔮 Next Day", 'fig_next', 'ax_next', 'canvas_next')
        
        # Results panel (Next Day Price, Accuracy, MAPE)
        results_frame = tk.Frame(right, bg=COLORS['bg_secondary'], relief='flat', bd=0)
        results_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=(8, 10))

        results_header = tk.Label(results_frame, text="RESULTS",
                      bg=COLORS['bg_secondary'], fg=COLORS['accent_green'],
                      font=('Segoe UI', 10, 'bold'))
        results_header.grid(row=0, column=0, sticky=tk.W, padx=(5, 15))

        # Next Day Price
        lbl_next = tk.Label(results_frame, text="Next Day:",
                    bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'],
                    font=('Segoe UI', 10))
        lbl_next.grid(row=0, column=1, sticky=tk.W, padx=(10, 5))
        val_next = tk.Label(results_frame, textvariable=self.result_nextday_var,
                    bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                    font=('Segoe UI', 10, 'bold'))
        val_next.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))

        # Accuracy
        lbl_acc = tk.Label(results_frame, text="Accuracy:",
                   bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'],
                   font=('Segoe UI', 10))
        lbl_acc.grid(row=0, column=3, sticky=tk.W, padx=(10, 5))
        val_acc = tk.Label(results_frame, textvariable=self.result_acc_var,
                   bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                   font=('Segoe UI', 10, 'bold'))
        val_acc.grid(row=0, column=4, sticky=tk.W, padx=(0, 20))

        # MAPE
        lbl_mape = tk.Label(results_frame, text="MAPE:",
                    bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'],
                    font=('Segoe UI', 10))
        lbl_mape.grid(row=0, column=5, sticky=tk.W, padx=(10, 5))
        val_mape = tk.Label(results_frame, textvariable=self.result_mape_var,
                    bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                    font=('Segoe UI', 10, 'bold'))
        val_mape.grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
    
    def _create_action_button(self, parent, text, command, color):
        """Create a styled action button"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=color, fg=COLORS['text_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat', borderwidth=0,
                       cursor='hand2', pady=12)
        
        # Hover effects
        def on_enter(e):
            btn.config(bg=self._darken_color(color))
        
        def on_leave(e):
            btn.config(bg=color)
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        
        return btn
    
    def _darken_color(self, hex_color, factor=0.8):
        """Darken a hex color"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(int(c * factor) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def _create_plot_tab(self, title, fig_name, ax_name, canvas_name):
        """Create a tab with a matplotlib plot"""
        fig = plt.Figure(figsize=(7, 5), facecolor="white")  # Set background to white
        ax = fig.add_subplot(111, facecolor="white")  # Set axes background to white
        ax.tick_params(colors=COLORS['text_secondary'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['top'].set_color(COLORS['border'])
        ax.spines['right'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
        
        setattr(self, fig_name, fig)
        setattr(self, ax_name, ax)
        
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        setattr(self, canvas_name, canvas)
        
        toolbar_frame = tk.Frame(tab, bg=COLORS['bg_tertiary'])
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)
    
    def _set_status(self, text):
        def update_status():
            self.status.config(text=text)
            if "Error" in text:
                self.status.config(fg=COLORS['accent_orange'])
            elif "complete" in text or "prepared" in text or "Ready" in text:
                self.status.config(fg=COLORS['accent_green'])
            else:
                self.status.config(fg=COLORS['accent_blue'])
        self.after(0, update_status)

    def _update_results(self, next_price, acc, mape):
        """Update the results panel values safely on the main thread."""
        try:
            if next_price is None:
                self.result_nextday_var.set("N/A")
            else:
                try:
                    self.result_nextday_var.set(f"${float(next_price):.2f}")
                except Exception:
                    self.result_nextday_var.set(str(next_price))

            if acc is None:
                self.result_acc_var.set("N/A")
            else:
                try:
                    self.result_acc_var.set(f"{float(acc):.2f}%")
                except Exception:
                    self.result_acc_var.set(str(acc))

            if mape is None:
                self.result_mape_var.set("N/A")
            else:
                try:
                    self.result_mape_var.set(f"{float(mape):.4f}")
                except Exception:
                    self.result_mape_var.set(str(mape))
        except Exception:
            # Keep previous values if update fails
            pass
    
    def _download_data(self):
        stock = self.selected_stock.get()
        self._set_status(f"Downloading {stock}...")
        
        def job():
            try:
                DownloadData(Config, stock)
                messagebox.showinfo("Success", f"Downloaded data for {stock}")
                self._set_status("Ready")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_status("Error")
        
        threading.Thread(target=job, daemon=True).start()
    
    def _load_and_prepare(self):
        stock = self.selected_stock.get()
        self._set_status(f"Loading data for {stock}...")
        
        def job():
            try:
                fp = GetStockFilePath(stock)
                data_date, data_close, num_points, date_range = FetchData(fp, Config)
                self.data_date = data_date
                self.data_close = data_close
                self.num_points = num_points
                
                try:
                    PlotRawData(self.data_date, self.data_close, self.num_points, date_range, Config, stock)
                except Exception:
                    pass
                
                self.scalar = Normalizer()
                normalized = self.scalar.FitTransformation(data_close)
                data_x, data_x_unseen = PrepDataX(normalized, PredictionCycle=Config["Data"]["PredictionCycle"])
                data_y = PrepDataY(normalized, PredictionCycle=Config["Data"]["PredictionCycle"])
                
                split_index = int(data_y.shape[0] * Config["Data"]["DataSplitRatio"])
                self.split_index = split_index
                t_x, te_x, t_y, te_y = SplitData(data_x, data_y, split_index)
                
                self.training_input = t_x
                self.testing_input = te_x
                self.training_output = t_y
                self.testing_output = te_y
                
                try:
                    Plot(DataDate=self.data_date,
                        TrainingData_Output=self.training_output,
                        TestingData_Output=self.testing_output,
                        Scalar=self.scalar,
                        NumDataPoints=self.num_points,
                        SplitIndex=self.split_index,
                        Config=Config,
                        Stock=stock,
                        mode="split")
                except Exception:
                    pass
                
                self._plot_split()
                
                try:
                    self._display_saved_plots(self.selected_stock.get())
                except Exception:
                    pass
                
                self._set_status("Data prepared")
                messagebox.showinfo("Done", "Data loaded and prepared")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_status("Error")
        
        threading.Thread(target=job, daemon=True).start()
    
    def _train_and_predict(self):
        model_name = self.selected_model.get()
        if self.data_close is None:
            messagebox.showwarning("No Data", "Please load and prepare data first")
            return
        
        self._set_status(f"Training {model_name}...")
        
        def job():
            try:
                if model_name == "LSTM":
                    model = LSTMModel(input_size=Config["Model"]["InputSize"],
                                      hidden_layer_size=Config["Model"]["LSTMSize"],
                                      num_layers=Config["Model"]["NumOfLSTMLayers"],
                                      output_size=1,
                                      dropout=Config["Model"]["Dropout"])
                    
                    trained = TrainModel(model, TimeSeriesDataset(self.training_input, self.training_output), 
                                        TimeSeriesDataset(self.testing_input, self.testing_output), Config)
                    train_preds, test_preds, mape, acc = EvaluateModel(trained, 
                                                                        TimeSeriesDataset(self.training_input, self.training_output), 
                                                                        TimeSeriesDataset(self.testing_input, self.testing_output), 
                                                                        self.scalar, Config)
                    next_pred = PredictNextDay(trained, self.testing_input, Config)
                    
                    try:
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=None,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="predicted",
                            DataClosePrice=self.data_close,
                            TrainingPredictions=train_preds,
                            TestingPredictions=test_preds)
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=self.testing_output,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="nextday",
                            TestingPredictions=test_preds,
                            Prediction=next_pred)
                    except Exception:
                        pass
                
                elif model_name == "XGBOOST":
                    trained = TrainXGBModel(self.training_input, self.training_output, 
                                           self.testing_input, self.testing_output)
                    train_preds, test_preds, mape, acc = EvaluateXGBModel(trained, self.training_input, 
                                                                          self.testing_input, self.testing_output, 
                                                                          self.scalar)
                    next_pred = PredictNextDayXGB(trained, self.testing_input[-1])
                    
                    try:
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=None,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="predicted",
                            DataClosePrice=self.data_close,
                            TrainingPredictions=train_preds,
                            TestingPredictions=test_preds)
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=self.testing_output,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="nextday",
                            TestingPredictions=test_preds,
                            Prediction=next_pred)
                    except Exception:
                        pass
                
                else:  # N-BEATS
                    trained = TrainNBeatsModel(TimeSeriesDataset(self.training_input, self.training_output), 
                                              TimeSeriesDataset(self.testing_input, self.testing_output), Config)
                    train_preds, test_preds, mape, acc = EvaluateNBeatsModel(trained, 
                                                                             TimeSeriesDataset(self.training_input, self.training_output), 
                                                                             TimeSeriesDataset(self.testing_input, self.testing_output), 
                                                                             self.scalar)
                    next_pred = PredictNextDayNBeats(trained, self.testing_input[-1])
                    
                    try:
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=None,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="predicted",
                            DataClosePrice=self.data_close,
                            TrainingPredictions=train_preds,
                            TestingPredictions=test_preds)
                        Plot(DataDate=self.data_date,
                            TrainingData_Output=None,
                            TestingData_Output=self.testing_output,
                            Scalar=self.scalar,
                            NumDataPoints=self.num_points,
                            SplitIndex=self.split_index,
                            Config=Config,
                            Stock=self.selected_stock.get(),
                            mode="nextday",
                            TestingPredictions=test_preds,
                            Prediction=next_pred)
                    except Exception:
                        pass
                
                def inv(array):
                    if self.scalar is None:
                        return array
                    if hasattr(self.scalar, "InverseTransformation"):
                        return self.scalar.InverseTransformation(array)
                    if hasattr(self.scalar, "InverseTransform"):
                        return self.scalar.InverseTransform(array)
                    return array
                
                self.training_preds = inv(train_preds)
                self.testing_preds = inv(test_preds)
                
                try:
                    val = inv(next_pred)
                    # If val is a numpy array or similar, extract a single element safely
                    if hasattr(val, 'item'):
                        self.prediction_next = float(val.item())
                    elif isinstance(val, (list, tuple)) and len(val) == 1:
                        self.prediction_next = float(val[0])
                    else:
                        self.prediction_next = float(val)
                except Exception:
                    self.prediction_next = next_pred
                
                self._plot_predicted()
                self._plot_nextday()
                
                try:
                    self._display_saved_plots(self.selected_stock.get())
                except Exception:
                    pass

                # Update results panel on the main thread
                try:
                    self.after(0, lambda: self._update_results(self.prediction_next, acc, mape))
                except Exception:
                    pass

                self._set_status("Training & Prediction complete")
                messagebox.showinfo("Done", "Training and prediction finished")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._set_status("Error")
        
        threading.Thread(target=job, daemon=True).start()
    
    def _plot_raw(self):
        self.ax_raw.clear()
        if self.data_date is None or self.data_close is None:
            self.ax_raw.text(0.5, 0.5, "No data", ha="center", color=COLORS['text_secondary'])
        else:
            self.ax_raw.plot(self.data_date, self.data_close, label="Close", color=COLORS['accent_blue'], linewidth=2)
            self.ax_raw.set_title(f"{self.selected_stock.get()} - Raw Close Prices", 
                                 color=COLORS['text_primary'], pad=15)
            self.ax_raw.legend(facecolor=COLORS['bg_tertiary'], edgecolor=COLORS['border'])
            self.ax_raw.grid(True, alpha=0.2, color=COLORS['border'])
        self.canvas_raw.draw()
    
    def _plot_split(self):
        self.ax_split.clear()
        if getattr(self, 'training_output', None) is None:
            self.ax_split.text(0.5, 0.5, "No split data", ha="center", color=COLORS['text_secondary'])
        else:
            try:
                train = self.scalar.InverseTransformation(self.training_output)
                test = self.scalar.InverseTransformation(self.testing_output)
            except Exception:
                train = self.training_output
                test = self.testing_output
            
            idx_train = range(len(train))
            idx_test = range(len(train), len(train) + len(test))
            self.ax_split.plot(idx_train, train, label="Train", color=COLORS['accent_green'], linewidth=2)
            self.ax_split.plot(idx_test, test, label="Test", color=COLORS['accent_orange'], linewidth=2)
            self.ax_split.set_title("Train/Test Split", color=COLORS['text_primary'], pad=15)
            self.ax_split.legend(facecolor=COLORS['bg_tertiary'], edgecolor=COLORS['border'])
            self.ax_split.grid(True, alpha=0.2, color=COLORS['border'])
        self.canvas_split.draw()
    
    def _plot_predicted(self):
        self.ax_pred.clear()
        if self.testing_preds is None:
            self.ax_pred.text(0.5, 0.5, "No predictions", ha="center", color=COLORS['text_secondary'])
        else:
            full = list(self.data_close)
            train_len = len(self.training_preds) if self.training_preds is not None else 0
            test_len = len(self.testing_preds) if self.testing_preds is not None else 0
            
            train_idx = range(train_len)
            test_idx = range(train_len, train_len + test_len)
            
            self.ax_pred.plot(full, label="Actual Close", color=COLORS['text_secondary'], alpha=0.7, linewidth=1.5)
            if self.training_preds is not None:
                self.ax_pred.plot(train_idx, self.training_preds, label="Train Preds", 
                                 color=COLORS['accent_green'], linewidth=2)
            if self.testing_preds is not None:
                self.ax_pred.plot(test_idx, self.testing_preds, label="Test Preds", 
                                 color=COLORS['accent_orange'], linewidth=2)
            
            self.ax_pred.set_title("Predictions vs Actual", color=COLORS['text_primary'], pad=15)
            self.ax_pred.legend(facecolor=COLORS['bg_tertiary'], edgecolor=COLORS['border'])
            self.ax_pred.grid(True, alpha=0.2, color=COLORS['border'])
        self.canvas_pred.draw()
    
    def _plot_nextday(self):
        self.ax_next.clear()
        if self.prediction_next is None:
            self.ax_next.text(0.5, 0.5, "No next-day prediction", ha="center", color=COLORS['text_secondary'])
        else:
            last_n = 50
            vals = list(self.data_close[-last_n:]) if self.data_close is not None else []
            xs = list(range(len(vals)))
            self.ax_next.plot(xs, vals, label="Recent Close", color=COLORS['accent_blue'], linewidth=2)
            self.ax_next.plot(len(vals), self.prediction_next, marker="o", markersize=12, 
                             color=COLORS['accent_orange'], label="Next Day Pred")
            self.ax_next.set_title("Next Day Prediction", color=COLORS['text_primary'], pad=15)
            self.ax_next.legend(facecolor=COLORS['bg_tertiary'], edgecolor=COLORS['border'])
            self.ax_next.grid(True, alpha=0.2, color=COLORS['border'])
        self.canvas_next.draw()
    
    def _display_saved_plots(self, stock: str):
        """
        Load saved plot images for `stock` from the project's PlotImages folder and display
        them on the existing axes (split, predicted, nextday). This prefers saved PNGs if
        present; otherwise leaves the current axes content.
        """
        project_dir = os.path.dirname(__file__)
        candidates = [os.path.join(project_dir, "PlotImages"), r"C:\Users\sense\Desktop\SEMESTER 7\Artificial Intelligence\Project\PlotImages"]
        
        files = {
            "raw": f"{stock}_raw.png",
            "split": f"{stock}_split.png",
            "predicted": f"{stock}_predicted.png",
            "nextday": f"{stock}_nextday.png",
        }
        
        mapping = [
            (files["raw"], self.ax_raw, self.canvas_raw),
            (files["split"], self.ax_split, self.canvas_split),
            (files["predicted"], self.ax_pred, self.canvas_pred),
            (files["nextday"], self.ax_next, self.canvas_next),
        ]
        
        for fname, ax, canvas in mapping:
            found = None
            for d in candidates:
                path = os.path.join(d, fname)
                if os.path.isfile(path):
                    found = path
                    break
            
            if found is None:
                rel = os.path.join(project_dir, "PlotImages", fname)
                if os.path.isfile(rel):
                    found = rel
            
            ax.clear()
            if found:
                try:
                    img = mpimg.imread(found)
                    ax.imshow(img)
                    ax.axis("off")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{os.path.basename(found)}", 
                           ha="center", color=COLORS['text_secondary'])
            else:
                ax.text(0.5, 0.5, f"PLEASE WAIT...:\n{fname}", 
                       ha="center", color=COLORS['text_secondary'])
            canvas.draw()
    
    def _cleanup_saved_plots(self):
        """
        Delete saved plot images for all stocks from the PlotImages folder.
        """
        project_dir = os.path.dirname(__file__)
        candidates = [os.path.join(project_dir, "PlotImages"), 
                     r"C:\Users\sense\Desktop\SEMESTER 7\Artificial Intelligence\Project\PlotImages"]
        
        for stock in STOCKS:
            files = [
                f"{stock}_raw.png",
                f"{stock}_split.png",
                f"{stock}_predicted.png",
                f"{stock}_nextday.png",
            ]
            
            for fname in files:
                for d in candidates:
                    path = os.path.join(d, fname)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            print(f"Error deleting file {path}: {e}")

    def _refresh(self):
        """
        Deletes any image files in the PlotImages folders and clears all plot axes
        displayed in the interface.
        """
        self._set_status("Refreshing plots...")

        project_dir = os.path.dirname(__file__)
        candidates = [os.path.join(project_dir, "PlotImages"),
                     r"C:\Users\sense\Desktop\SEMESTER 7\Artificial Intelligence\Project\PlotImages"]

        # Remove image files in candidate directories
        removed = 0
        for d in candidates:
            try:
                if os.path.isdir(d):
                    for fname in os.listdir(d):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif')):
                            path = os.path.join(d, fname)
                            try:
                                os.remove(path)
                                removed += 1
                            except Exception:
                                pass
            except Exception:
                pass

        # Clear all axes and redraw empty placeholders
        try:
            for ax, canvas in ((self.ax_raw, self.canvas_raw),
                               (self.ax_split, self.canvas_split),
                               (self.ax_pred, self.canvas_pred),
                               (self.ax_next, self.canvas_next)):
                try:
                    ax.clear()
                    ax.text(0.5, 0.5, "No plot available", ha="center", color=COLORS['text_secondary'])
                    if hasattr(canvas, 'draw'):
                        canvas.draw()
                except Exception:
                    pass
        except Exception:
            pass

        # Also reset the results display so the UI fully refreshes
        try:
            self.result_nextday_var.set("N/A")
            self.result_acc_var.set("N/A")
            self.result_mape_var.set("N/A")
        except Exception:
            pass

        self._set_status(f"Refresh complete ({removed} files removed)")
    
    def destroy(self):
        """
        Override destroy to ensure cleanup is called.
        """
        self._cleanup_saved_plots()
        super().destroy()


def run_app():
    app = StockPredictorUI()
    app.mainloop()


if __name__ == "__main__":
    run_app()