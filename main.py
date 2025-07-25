import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

# === STEP 1: Download historical data ===
ticker = 'AAPL'
print(f"📡 Downloading historical data for {ticker}...")

# Try different approaches to handle rate limiting
try:
    # First, try downloading with a longer period to avoid rate limits
    data = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)
    
    if data.empty:
        print("⚠️ Data download failed. Trying alternative approach...")
        # Fallback: try a shorter period
        data = yf.download(ticker, period='5y', progress=False)
    
    if data.empty:
        raise ValueError("Unable to download stock data. Please check your internet connection or try again later.")
    
    # Clean the data
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data: only {len(data)} rows available. Need at least 100 rows for reliable predictions.")
    
    print(f"✅ Successfully downloaded {len(data)} days of data from {data.index[0].date()} to {data.index[-1].date()}")
    
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    print("💡 Tip: Try running the script again in a few minutes if you're rate limited.")
    exit(1)

# === STEP 2: Calculate indicators ===
# Calculate indicators on the full dataset to ensure consistency
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# Add more technical indicators for better Random Forest performance
data['RSI'] = calculate_rsi(data['Close'])
data['BB_upper'], data['BB_lower'] = calculate_bollinger_bands(data['Close'])
data['Price_Change'] = data['Close'].pct_change()
data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
data['High_Low_Ratio'] = data['High'] / data['Low']
data['Close_Open_Ratio'] = data['Close'] / data['Open']

# === STEP 3: Create target and features ===
# The target is the next day's closing price
data['Target'] = data['Close'].shift(-1)

# Keep the last row for prediction before dropping NaNs
# This row has the most recent features to predict the next day's price
latest_data_for_prediction = data.iloc[-1:].copy()

# Drop rows with NaN values (especially the last row where Target is NaN) for training
data_clean = data.dropna()

# Check if we have enough data after cleaning
if len(data_clean) == 0:
    print("❌ Error: No valid data available after cleaning. All rows contain NaN values.")
    print("💡 This might be due to insufficient data or calculation errors.")
    exit(1)

if len(data_clean) < 100:
    print(f"⚠️ Warning: Only {len(data_clean)} rows of clean data available.")
    print("💡 Consider using a longer time period for more reliable predictions.")

# Use a more comprehensive feature set for better model performance
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_10', 'EMA_50', 'SMA_10', 'SMA_50', 
           'MACD', 'RSI', 'BB_upper', 'BB_lower', 'Price_Change', 'Volume_MA', 
           'High_Low_Ratio', 'Close_Open_Ratio']
X = data_clean[features]
y = data_clean['Target']

print(f"📊 Training dataset shape: {X.shape}")
print(f"📊 Features: {features}")

# Scale features for better Random Forest performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)

# Save the scaler for later use
joblib.dump(scaler, 'feature_scaler.pkl')

# === STEP 4: Train/test split ===
# Split data into training and testing sets, ensuring order is maintained
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# === STEP 5: Train models ===
print("\n🤖 Training models...")

# Linear Regression (works well with scaled features)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'linear_model.pkl')

# Try Random Forest without scaling - it might work better with raw features
X_train_unscaled, X_test_unscaled, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

# Improved Random Forest with different hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=50,            # Fewer trees to prevent overfitting
    max_depth=10,               # Shallower trees
    min_samples_split=10,       # More conservative splitting
    min_samples_leaf=5,         # Larger leaf size
    max_features='log2',        # Different feature selection
    bootstrap=True,             # Use bootstrap sampling
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_unscaled, y_train)
joblib.dump(rf_model, 'rf_model.pkl')

print("✅ Models trained successfully!")

# === STEP 6: Prepare latest data for prediction ===
# Load the scaler and scale the latest features for Linear Regression
scaler_loaded = joblib.load('feature_scaler.pkl')
latest_features = latest_data_for_prediction[features].values
latest_features_scaled = scaler_loaded.transform(latest_features)

# For Random Forest, use unscaled features
latest_features_unscaled = latest_features

# Print latest feature values
print("\n📊 Latest Feature Values for Prediction:")
print(latest_data_for_prediction[features])

# === STEP 7: Load models and predict ===
lr_model_loaded = joblib.load('linear_model.pkl')
rf_model_loaded = joblib.load('rf_model.pkl')

lr_pred = lr_model_loaded.predict(latest_features_scaled)[0]
rf_pred = rf_model_loaded.predict(latest_features_unscaled)[0]

# === STEP 8: Show predictions ===
current_price = float(latest_data_for_prediction['Close'].iloc[0])
print(f"\n💰 Current Price: ${current_price:.2f}")
print("\n📈 Predictions for Next Day's Close Price")
print(f"Linear Regression Prediction: ${lr_pred:.2f} ({((lr_pred/current_price - 1)*100):+.2f}%)")
print(f"Random Forest Prediction:    ${rf_pred:.2f} ({((rf_pred/current_price - 1)*100):+.2f}%)")

# Calculate ensemble prediction
ensemble_pred = (lr_pred + rf_pred) / 2
print(f"Ensemble Prediction:         ${ensemble_pred:.2f} ({((ensemble_pred/current_price - 1)*100):+.2f}%)")

# === STEP 9: Evaluate model performance on the test set ===
lr_test_pred = lr_model.predict(X_test)
rf_test_pred = rf_model.predict(X_test_unscaled)

lr_r2 = r2_score(y_test, lr_test_pred)
rf_r2 = r2_score(y_test, rf_test_pred)
lr_mae = mean_absolute_error(y_test, lr_test_pred)
rf_mae = mean_absolute_error(y_test, rf_test_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))

print("\n📉 Model Performance on Test Set")
print("="*50)
print(f"Linear Regression:")
print(f"  R² Score: {lr_r2:.4f}")
print(f"  MAE:      ${lr_mae:.2f}")
print(f"  RMSE:     ${lr_rmse:.2f}")
print(f"\nRandom Forest:")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  MAE:      ${rf_mae:.2f}")
print(f"  RMSE:     ${rf_rmse:.2f}")

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top 5 Most Important Features for Random Forest:")
print(feature_importance.head().to_string(index=False))

# === STEP 10: Plot actual vs predicted prices on the test set ===
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Close', alpha=0.7, linewidth=2)
plt.plot(y_test.index, lr_test_pred, label='Linear Regression Prediction', linestyle='--', alpha=0.8)
plt.plot(y_test.index, rf_test_pred, label='Random Forest Prediction', linestyle=':', alpha=0.8)
plt.title(f"{ticker} Stock Price: Actual vs. Predicted (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()