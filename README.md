# Stock Price Prediction using Random Forest Classifier

A machine learning project that uses Random Forest Classification to predict stock price movements based on technical indicators. The model analyzes historical stock data and predicts whether the price will increase or decrease over different time horizons.

## Features

- Fetches historical stock data using Yahoo Finance API
- Implements multiple technical indicators:
  - Relative Strength Index (RSI)
  - Stochastic Oscillator
  - Williams %R
  - Moving Average Convergence Divergence (MACD)
  - Rate of Change (ROC)
  - On-Balance Volume (OBV)
- Uses exponential smoothing for price data
- Performs time series cross-validation
- Implements grid search for hyperparameter optimization
- Generates ROC curves and calculates AUC scores
- Supports multiple prediction windows (30, 60, and 90 days)

## Requirements

```
python
pandas
numpy
scikit-learn
yfinance
seaborn
matplotlib
```

## Technical Indicators

The project calculates several technical indicators to predict stock price movements:
- **RSI**: Measures momentum by comparing the magnitude of recent gains to recent losses
- **Stochastic Oscillator**: Shows momentum by comparing a closing price to its price range
- **Williams %R**: Momentum indicator showing overbought and oversold levels
- **MACD**: Trend-following momentum indicator showing relationship between two moving averages
- **ROC**: Shows the speed at which price is changing
- **OBV**: Volume-based indicator for analyzing price and volume trends

## Model Performance

The Random Forest Classifier shows strong performance across different stocks and prediction windows:

### AAPL
- 30-day prediction: AUC = 0.85, OOB Error = 0.33
- 60-day prediction: AUC = 0.87, OOB Error = 0.28
- 90-day prediction: AUC = 0.89, OOB Error = 0.24

### GE
- 30-day prediction: AUC = 0.87, OOB Error = 0.35
- 60-day prediction: AUC = 0.89, OOB Error = 0.31
- 90-day prediction: AUC = 0.99, OOB Error = 0.26

### Samsung (005930.KS)
- 30-day prediction: AUC = 0.97, OOB Error = 0.35
- 60-day prediction: AUC = 0.86, OOB Error = 0.36
- 90-day prediction: AUC = 0.97, OOB Error = 0.31

## Model Configuration

The Random Forest Classifier uses the following optimized parameters:
- Number of estimators: 45
- Maximum depth: 12
- Bootstrap: True
- OOB Score: Enabled
- Random State: 0

## Usage

1. Import the required libraries
2. Use `fetch_stock_data()` to get historical data
3. Calculate technical indicators using `calculate_technical_indicators()`
4. Create binary labels using `create_labels()`
5. Prepare data using `prepare_data()`
6. Train and evaluate the model using `train_and_evaluate_model()`

## Visualization

The project includes several visualization tools:
- Correlation heatmaps of features
- Feature distribution histograms
- ROC curves for model performance
- Class distribution plots

## Notes

- The model uses time series cross-validation to prevent look-ahead bias
- Out-of-bag (OOB) error is used to assess model performance
- Performance generally improves with longer prediction windows
- The model achieves better AUC scores for longer-term predictions
