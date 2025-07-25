# Stock Price Prediction Using Machine Learning

A Python-based stock price prediction system that uses Linear Regression and Random Forest models to forecast next-day closing prices for stocks. The system incorporates multiple technical indicators and provides comprehensive model evaluation.

## 🚀 Features

- **Real-time Data**: Downloads historical stock data using Yahoo Finance API
- **Technical Indicators**: Calculates multiple technical indicators including:
  - Exponential Moving Averages (EMA 10, 50)
  - Simple Moving Averages (SMA 10, 50)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Price change percentages
  - Volume moving averages
  - Price ratios
- **Dual Model Approach**: Uses both Linear Regression and Random Forest models
- **Ensemble Prediction**: Combines predictions from both models
- **Comprehensive Evaluation**: Provides R², MAE, and RMSE metrics
- **Visualization**: Plots actual vs predicted prices
- **Error Handling**: Robust handling of API rate limits and data issues

## 📋 Requirements

```
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
```

## 🛠️ Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd stock_price_prediction
   ```
3. Install required packages:
   ```bash
   pip install yfinance pandas numpy matplotlib scikit-learn
   ```

## 📊 Usage

### Basic Usage

Run the main script to predict Apple (AAPL) stock prices:

```bash
python main.py
```

### Changing the Stock Symbol

To predict prices for a different stock, modify the `ticker` variable in `main.py`:

```python
ticker = 'MSFT'  # For Microsoft
ticker = 'GOOGL' # For Google
ticker = 'TSLA'  # For Tesla
```

### Output

The script provides:

1. **Current Price**: Latest closing price
2. **Predictions**: Next-day price predictions from both models
3. **Performance Metrics**: R², MAE, and RMSE for both models
4. **Feature Importance**: Top features used by the Random Forest model
5. **Visualization**: Graph comparing actual vs predicted prices

Example output:
```
💰 Current Price: $251.59

📈 Predictions for Next Day's Close Price
Linear Regression Prediction: $252.10 (+0.20%)
Random Forest Prediction:    $172.18 (-31.56%)
Ensemble Prediction:         $212.14 (-15.68%)

📉 Model Performance on Test Set
==================================================
Linear Regression:
  R² Score: 0.9909
  MAE:      $1.91
  RMSE:     $2.59

Random Forest:
  R² Score: -0.3053
  MAE:      $21.27
  RMSE:     $31.03
```

## 🔧 Technical Details

### Data Processing

1. **Data Download**: Fetches 10 years of historical data (2015-2024)
2. **Feature Engineering**: Calculates 17 technical indicators
3. **Data Cleaning**: Removes rows with missing values
4. **Feature Scaling**: Applies StandardScaler for Linear Regression

### Models

#### Linear Regression
- Uses scaled features for better performance
- Excellent for capturing linear relationships
- Generally provides more stable predictions

#### Random Forest
- Uses unscaled features (works better with raw data)
- Captures non-linear relationships
- Provides feature importance rankings
- May be sensitive to overfitting with financial data

### Features Used

| Feature | Description |
|---------|-------------|
| Open, High, Low, Close | Basic OHLC data |
| Volume | Trading volume |
| EMA_10, EMA_50 | Exponential moving averages |
| SMA_10, SMA_50 | Simple moving averages |
| MACD | Moving average convergence divergence |
| RSI | Relative strength index |
| BB_upper, BB_lower | Bollinger bands |
| Price_Change | Daily price change percentage |
| Volume_MA | Volume moving average |
| High_Low_Ratio | High/Low price ratio |
| Close_Open_Ratio | Close/Open price ratio |

## 📈 Model Performance

### Linear Regression
- **Strengths**: Stable, interpretable, good R² scores
- **Weaknesses**: Limited to linear relationships

### Random Forest
- **Strengths**: Captures complex patterns, provides feature importance
- **Weaknesses**: Can overfit, may struggle with time series data

## ⚠️ Important Notes

### Limitations
- **Financial markets are unpredictable**: Past performance doesn't guarantee future results
- **Random Forest issues**: May show negative R² scores due to overfitting
- **Data dependency**: Requires stable internet connection for data download
- **Rate limiting**: Yahoo Finance may limit API requests

### Risk Disclaimer
**This tool is for educational purposes only. Do not use these predictions for actual trading decisions without proper financial analysis and risk management.**

## 🔧 Troubleshooting

### Common Issues

1. **Rate Limiting Error**:
   ```
   YFRateLimitError: Too Many Requests
   ```
   **Solution**: Wait a few minutes and try again

2. **Empty Dataset Error**:
   ```
   ValueError: With n_samples=0, test_size=0.2
   ```
   **Solution**: Check internet connection and try a different stock symbol

3. **Poor Random Forest Performance**:
   - This is common with financial time series data
   - The Linear Regression model typically performs better
   - Consider using only the Linear Regression predictions

## 📁 Generated Files

The script creates several files:
- `linear_model.pkl`: Trained Linear Regression model
- `rf_model.pkl`: Trained Random Forest model
- `feature_scaler.pkl`: StandardScaler for feature normalization

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:
- Add more technical indicators
- Implement additional ML models (LSTM, XGBoost, etc.)
- Improve error handling
- Add more visualization options

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Ensure all dependencies are installed
3. Verify internet connection for data download
4. Try with a different stock symbol

---

**Remember**: Stock trading involves significant risk. This tool is for educational purposes only and should not be used as the sole basis for investment decisions.
