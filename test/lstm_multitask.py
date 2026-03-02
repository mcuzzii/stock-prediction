import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
import tensorflow as tf
import os

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
sns.set_style('whitegrid')
tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = 'data/processed/ohlc_df.csv'
SEQ_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 20 # Increased slightly to ensure convergence
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# -----------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# -----------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Handle datetime
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Drop rows with NaNs immediately to clean start
df.dropna(inplace=True)

print(f"Data Loaded. Shape: {df.shape}")

# -----------------------------------------------------------------------------
# 3. Feature Engineering
# -----------------------------------------------------------------------------
print("Creating Time & volatility features...")

# User Requirement: "scalas for every single stock price"
# We will identify stock families and create features per stock
stock_prefixes = set([c.split('_')[0] for c in df.columns if '_' in c])
print(f"Identified {len(stock_prefixes)} stocks.")

# 3.1 Time Features
# We work on the index directly
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['dayofweek'] = df.index.dayofweek

# 3.2 Market Phases (Heuristic based on PSE typical hours or inferred)
# Assuming 9:30 Start, 12:00-1:00 Recess, 3:00 Close.
# Adjust logic: "near recess", "market open/close"

def get_market_flags(row):
    # Time in minutes from start of day (09:30)
    t_min = row.hour * 60 + row.minute
    
    # Flags
    is_open = 0
    is_close = 0
    is_recess = 0
    
    # Open: First 15 mins (9:30 - 9:45) -> 570 - 585
    if 570 <= t_min <= 585:
        is_open = 1
        
    # Recess: Around 12:00 (11:45 - 12:00 for pre-recess?) -> 705 - 720
    # Data might just stop. Let's mark the "last valid candle before 12" as recess-near.
    # Simple logic: 11:30 - 12:00
    if 690 <= t_min <= 720:
        is_recess = 1
    # 1:00 - 1:30 (Post recess volatility) -> 780 - 810
    if 780 <= t_min <= 810:
        is_recess = 1
        
    # Close: Last 30 mins (2:30 PM - 3:00 PM) -> 14:30 - 15:00 -> 870 - 900
    if 870 <= t_min <= 900:
        is_close = 1
        
    return pd.Series([is_open, is_recess, is_close])

df[['flag_open', 'flag_recess', 'flag_close']] = df.apply(get_market_flags, axis=1)

# 3.3 Volatility Features
# Calculate Rolling Volatility for every stock's return
for prefix in stock_prefixes:
    col = f'{prefix}_perc_chg'
    if col in df.columns:
        # 20-period rolling std
        df[f'{prefix}_volatility'] = df[col].rolling(window=20).std()

df.dropna(inplace=True)

# -----------------------------------------------------------------------------
# 4. Target & Feature Definition
# -----------------------------------------------------------------------------
# Targets: All '_perc_chg' columns
target_cols = [c for c in df.columns if c.endswith('_perc_chg')]
n_stocks = len(target_cols)

# Features: All numeric columns
# We scale EVERYTHING.
# User Logic: "scalers for every single stock price".
# We will use scaler objects.

feature_cols = df.columns.tolist()
# Ensure targets are also in features (autoregressive)
print(f"Total Features: {len(feature_cols)}, Total Targets: {len(target_cols)}")

# -----------------------------------------------------------------------------
# 5. Split & Scale
# -----------------------------------------------------------------------------
train_size = int(len(df) * (1 - TEST_SPLIT - VAL_SPLIT))
val_size = int(len(df) * VAL_SPLIT)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

# Scalers
# Use RobustScaler for features to handle price spikes? Or Standard.
# User asked to "learn noise proportionately". Standard scaler preserves distribution shape better than MinMax.
scaler_X = StandardScaler()
# IMPORTANT: "Center at Zero" -> with_mean=False for returns might be good, 
# but StandardScaler(with_mean=True) centers EXACTLY at zero which is what we want for inputs.
# For TARGETS (Returns), we essentially want 0% return to be 0.
# If we mean-center returns, we imply the mean return is 0 (which is roughly true for high freq).

# However, user wants model to REMAIN centered at zero.
# If we fit scaler_y with mean=True, then 0 input -> Mean Return.
# If we fit scaler_y with mean=False, then 0 input -> 0 Return.
# We choose mean=False for Targets to preserve the "Zero Return" anchor.
scaler_y = StandardScaler(with_mean=False) 

# Fit Scalers
X_train_raw = train_df[feature_cols].values
y_train_raw = train_df[target_cols].values

scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

# Transform
X_train = scaler_X.transform(train_df[feature_cols].values)
y_train = scaler_y.transform(train_df[target_cols].values)

X_val = scaler_X.transform(val_df[feature_cols].values)
y_val = scaler_y.transform(val_df[target_cols].values)

X_test = scaler_X.transform(test_df[feature_cols].values)
y_test = scaler_y.transform(test_df[target_cols].values)

# -----------------------------------------------------------------------------
# 6. Sequence Creation (Sliding Window)
# -----------------------------------------------------------------------------
# Multi-output: Input (N, T, F) -> Output (N, n_stocks)
# Hard Parameter Sharing: One model, many outputs.

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

print("Creating sequences...")
X_train_seq, y_train_seq = create_dataset(X_train, y_train, SEQ_LENGTH)
X_val_seq, y_val_seq = create_dataset(X_val, y_val, SEQ_LENGTH)
X_test_seq, y_test_seq = create_dataset(X_test, y_test, SEQ_LENGTH)

print(f"Train Seq Shape: {X_train_seq.shape} -> {y_train_seq.shape}")

# -----------------------------------------------------------------------------
# 7. Model Building (Hard Parameter Sharing)
# -----------------------------------------------------------------------------
# Shared Bottom
input_layer = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

# LSTM Layers (Shared)
x = LSTM(128, return_sequences=True)(input_layer)
# Minimal dropout to allow learning "noise" / volatility details
x = Dropout(0.1)(x) 
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.1)(x)

# Shared Dense info bottleneck
x = Dense(64, activation='relu')(x)

# Output Layer
# We predict n_stocks returns simultaneously.
output_layer = Dense(n_stocks)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Loss: Huber Loss is robust to outliers but quadratic near zero (good for noise).
# User wants "learn noise proportionately".
model.compile(optimizer='adam', loss='huber', metrics=['mae', 'mse'])

model.summary()

# -----------------------------------------------------------------------------
# 8. Training
# -----------------------------------------------------------------------------
print("Starting Training...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_seq, y_val_seq),
    shuffle=True,
    verbose=1
)

# -----------------------------------------------------------------------------
# 9. Evaluation
# -----------------------------------------------------------------------------
print("Evaluating...")

# Predict
y_pred_scaled = model.predict(X_test_seq)

# Inverse Transform
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_seq)

# Metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"Overall Mean Squared Error: {mse}")
print(f"Overall Mean Absolute Error: {mae}")

# Per Stock Metrics
rmse_per_stock = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
results = pd.DataFrame({
    'Stock': target_cols,
    'RMSE': rmse_per_stock
}).sort_values('RMSE')

print("Top 5 Best Predicted Stocks (Lowest RMSE):")
print(results.head())

print("Top 5 Worst Predicted Stocks (Highest RMSE):")
print(results.tail())

# -----------------------------------------------------------------------------
# 10. Visualization (One Example)
# -----------------------------------------------------------------------------
plt.figure(figsize=(15, 6))
# Pick the most volatile stock or first one
stock_idx = 0
stock_name = target_cols[stock_idx]

plt.plot(y_true[:, stock_idx], label='Actual', alpha=0.7)
plt.plot(y_pred[:, stock_idx], label='Predicted', alpha=0.7)
plt.title(f'Multi-Output LSTM Prediction: {stock_name}')
plt.legend()
plt.tight_layout()
plt.savefig('multitask_prediction_example.png')
print("Saved plot to multitask_prediction_example.png")

# Also plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss (Huber)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('multitask_loss.png')
print("Saved loss plot to multitask_loss.png")
