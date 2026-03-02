
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Set plot style
sns.set_style('whitegrid')

# Load data
print("Loading data...")
ohlc_df = pd.read_csv('data/processed/ohlc_df.csv')

# Preprocessing
# Rename the first column to 'datetime' and convert to datetime objects
if 'Unnamed: 0' in ohlc_df.columns:
    ohlc_df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
ohlc_df['datetime'] = pd.to_datetime(ohlc_df['datetime'])
ohlc_df.set_index('datetime', inplace=True)

# Feature Engineering: Rolling Volatility (Fix for volatile periods)
print("Adding volatility features...")
qt = 20
for col in ohlc_df.columns:
    if col.endswith('_perc_chg'):
        # Calculate rolling standard deviation (volatility)
        stock_name = col.replace('_perc_chg', '')
        ohlc_df[f'{stock_name}_volatility'] = ohlc_df[col].rolling(window=qt).std()

# Clean data: Drop rows with NaNs
ohlc_df.dropna(inplace=True)

# Identify targets and features
target_cols = [col for col in ohlc_df.columns if col.endswith('_perc_chg')]
# Use all numeric columns as features (now includes volatility)
feature_cols = ohlc_df.columns.tolist()

print(f"Target Columns ({len(target_cols)}): {target_cols}")
print(f"Total Features: {len(feature_cols)}")

data = ohlc_df[feature_cols].copy()

# 1. Split Data
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_df = data.iloc[:train_size]
val_df = data.iloc[train_size:train_size+val_size]
test_df = data.iloc[train_size+val_size:]

# 2. Scale Data
# Scaler for Features (X)
scaler_X = StandardScaler()
# Scaler for Targets (y)
# Fix for "offset from zero": Do not center the target (assume 0 mean for returns)
scaler_y = StandardScaler(with_mean=False)

# Fit on train data
X_train_raw = train_df.values
y_train_raw = train_df[target_cols].values

scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

# Transform
X_train_scaled = scaler_X.transform(train_df.values)
y_train_scaled = scaler_y.transform(train_df[target_cols].values)

X_val_scaled = scaler_X.transform(val_df.values)
y_val_scaled = scaler_y.transform(val_df[target_cols].values)

X_test_scaled = scaler_X.transform(test_df.values)
y_test_scaled = scaler_y.transform(test_df[target_cols].values)

print("Scalers fit complete.")

# Create sequences
def create_sequences(X_data, y_data, seq_length):
    X, y = [], []
    for i in range(len(X_data) - seq_length):
        # Input Sequence: Steps [i] to [i + seq_length - 1]
        X.append(X_data[i:i + seq_length])
        # Target: Step [i + seq_length] ONLY (Single step prediction)
        y.append(y_data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)
X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, SEQ_LENGTH)

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Validation shape: {X_val.shape}, {y_val.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

# Build Model (Task Conditioned approach from notebook)
# We augment the data to train a single-output LSTM that can predict any stock given a task ID.

# Optimize Data Creation using a Generator to satisfy memory constraints
class TaskDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_data, y_data, batch_size=128, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples, self.seq_length, self.n_features = X_data.shape
        self.n_stocks = y_data.shape[1]
        self.indexes = np.arange(self.n_samples * self.n_stocks)

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.zeros((len(batch_indexes), self.seq_length, self.n_features + self.n_stocks))
        y = np.zeros((len(batch_indexes)))

        for i, idx_val in enumerate(batch_indexes):
            sample_idx = idx_val // self.n_stocks
            stock_idx = idx_val % self.n_stocks
            X[i, :, :self.n_features] = self.X_data[sample_idx]
            X[i, :, self.n_features + stock_idx] = 1.0
            y[i] = self.y_data[sample_idx, stock_idx]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

print("Initializing Data Generators...")
train_gen = TaskDataGenerator(X_train, y_train, batch_size=128, shuffle=True)
val_gen = TaskDataGenerator(X_val, y_val, batch_size=128, shuffle=False)

# Determine input shape from data
n_features = X_train.shape[2]
n_stocks = y_train.shape[1]
input_shape = (SEQ_LENGTH, n_features + n_stocks)

print(f"Model Input Shape: {input_shape}")

# Model
# Remove Dropout to allow model to overfit/learn noise better
model_task = Sequential([
    LSTM(128, return_sequences=True, input_shape=input_shape),
    LSTM(64, return_sequences=False),
    Dense(1) # Single output
])

model_task.compile(optimizer='adam', loss='mean_absolute_error')
model_task.summary()

# Train
history_task = model_task.fit(
    train_gen,
    epochs=10, 
    validation_data=val_gen,
    verbose=1
)

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(history_task.history['loss'], label='Train Loss')
plt.plot(history_task.history['val_loss'], label='Validation Loss')
plt.title('Task-Conditioned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('task_model_loss.png')
print("Loss plot saved to task_model_loss.png")

# Evaluation
print("Evaluating...")
# Predictions are done per-stock below to avoid memory issues
# predictions_task = model_task.predict(X_test_task) # Removed to save memory

# We need to reshape predictions back to (n_samples, n_stocks) to compare effectively
# OR we evaluate on the long form.
# The user's notebook did inverse transform.
# Since we mixed everything, inverse transform is tricky unless we know which prediction belongs to which stock.
# BUTscaler_y fits on all columns. if we transform a single column, it expects shape (n, n_stocks).
# Actually scaler_y expects (n, n_stocks).
# We can't inverse transform the 1D array directly if we want to separate stocks.

# Let's perform per-stock evaluation using the un-shuffled test set logic
n_stocks = len(target_cols)
n_test_samples = X_test.shape[0]

# Pre-allocate array for all predictions
all_preds = np.zeros((n_test_samples, n_stocks))

for stock_idx in range(n_stocks):
    # Prepare input for this stock
    X_copy = X_test.copy()
    task_encoding = np.zeros((n_test_samples, SEQ_LENGTH, n_stocks))
    task_encoding[:, :, stock_idx] = 1.0
    X_cond = np.concatenate([X_copy, task_encoding], axis=2)
    
    # Predict
    preds = model_task.predict(X_cond, verbose=0)
    all_preds[:, stock_idx] = preds.flatten()

# Inverse transform
inv_predictions = scaler_y.inverse_transform(all_preds)
inv_actual = scaler_y.inverse_transform(y_test)

# Calculate RMSE
rmse_per_col = np.sqrt(np.mean((inv_actual - inv_predictions)**2, axis=0))
mean_rmse = np.mean(rmse_per_col)

print(f"Mean RMSE across all {len(target_cols)} stocks: {mean_rmse}")
results_df = pd.DataFrame({'Stock': target_cols, 'RMSE': rmse_per_col})
print(results_df.sort_values('RMSE').head())

# Plot Example
target_idx = 0
stock_name = target_cols[target_idx]

plt.figure(figsize=(16, 6))
plt.plot(inv_actual[:, target_idx], label=f'Actual {stock_name}', alpha=0.7)
plt.plot(inv_predictions[:, target_idx], label=f'Predicted {stock_name}', alpha=0.7)
plt.title(f'Prediction for {stock_name} (Fixed Offset & Volatility)')
plt.legend()
plt.savefig('prediction_example.png')
print("Prediction plot saved to prediction_example.png")
