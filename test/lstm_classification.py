import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Reshape, Softmax
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
sns.set_style('whitegrid')
tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = 'test/data/processed/ohlc_df.csv'
SEQ_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 30 
EPS = 0.0001
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# -----------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# -----------------------------------------------------------------------------
print("Loading data...")
if not os.path.exists(DATA_PATH):
    # Fallback for different execution contexts
    DATA_PATH = 'data/processed/ohlc_df.csv'

df = pd.read_csv(DATA_PATH)

# Handle datetime
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Drop rows with NaNs immediately
df.dropna(inplace=True)

print(f"Data Loaded. Shape: {df.shape}")

# -----------------------------------------------------------------------------
# 3. Feature Engineering
# -----------------------------------------------------------------------------
print("Creating Time & volatility features...")

stock_prefixes = set([c.split('_')[0] for c in df.columns if '_' in c])
print(f"Identified {len(stock_prefixes)} stocks.")

# 3.1 Time Features
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['dayofweek'] = df.index.dayofweek

# 3.2 Market Phases (Vectorized)
t_min = df['hour'] * 60 + df['minute']
df['flag_open'] = ((t_min >= 570) & (t_min <= 585)).astype(int)
df['flag_recess'] = (((t_min >= 690) & (t_min <= 720)) | ((t_min >= 780) & (t_min <= 810))).astype(int)
df['flag_close'] = ((t_min >= 870) & (t_min <= 900)).astype(int)

# 3.3 Volatility Features
for prefix in stock_prefixes:
    col = f'{prefix}_perc_chg'
    if col in df.columns:
        df[f'{prefix}_volatility'] = df[col].rolling(window=20).std()

df.dropna(inplace=True)

# -----------------------------------------------------------------------------
# 4. Classification Targets & Feature Definition
# -----------------------------------------------------------------------------
# Targets: Direction indicators (1 if > 0, 0 otherwise)
target_cols = [c for c in df.columns if c.endswith('_perc_chg')]
n_stocks = len(target_cols)

# Create 3-class labels: 0=Negative, 1=Neutral, 2=Positive
label_cols = []
all_labels_flat = []
for col in target_cols:
    label_name = col.replace('_perc_chg', '_class')
    
    # 0: Neg ( < -EPS ), 1: Neu ( abs <= EPS ), 2: Pos ( > EPS )
    df[label_name] = 1 # Default to Neutral
    df.loc[df[col] < -EPS, label_name] = 0
    df.loc[df[col] > EPS, label_name] = 2
    
    label_cols.append(label_name)
    all_labels_flat.extend(df[label_name].values)

# --- Visualization of Class Distribution ---
plt.figure(figsize=(8, 6))
sns.countplot(x=all_labels_flat, palette='viridis')
plt.title('Distribution of Classes (All Stocks Combined)')
plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
plt.ylabel('Count')
plt.savefig('class_distribution.png')
print("Saved class distribution plot to: class_distribution.png")

# Features: All numeric columns (excluding labels)
feature_cols = [c for c in df.columns if not c.endswith('_class')]
print(f"Total Features: {len(feature_cols)}, Total Stocks: {n_stocks}")

# -----------------------------------------------------------------------------
# 5. Split & Scale
# -----------------------------------------------------------------------------
train_size = int(len(df) * (1 - TEST_SPLIT - VAL_SPLIT))
val_size = int(len(df) * VAL_SPLIT)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

# Scaler for Features
scaler_X = StandardScaler()
scaler_X.fit(train_df[feature_cols].values)

X_train = scaler_X.transform(train_df[feature_cols].values)
X_val = scaler_X.transform(val_df[feature_cols].values)
X_test = scaler_X.transform(test_df[feature_cols].values)

# Targets (One-hot encode each stock's labels)
# y data currently has shape (N, n_stocks) with values 0, 1, 2
y_train_raw = train_df[label_cols].values
y_val_raw = val_df[label_cols].values
y_test_raw = test_df[label_cols].values

def one_hot_multi_output(y_array, num_classes=3):
    # Input y_array shape: (N, n_stocks)
    # Output shape: (N, n_stocks, num_classes)
    N, n_s = y_array.shape
    y_oh = np.zeros((N, n_s, num_classes))
    for i in range(n_s):
        y_oh[:, i, :] = to_categorical(y_array[:, i], num_classes=num_classes)
    return y_oh

y_train = one_hot_multi_output(y_train_raw)
y_val = one_hot_multi_output(y_val_raw)
y_test = one_hot_multi_output(y_test_raw)

print(f"Target shapes: train={y_train.shape}, val={y_val.shape}, test={y_test.shape}")

# -----------------------------------------------------------------------------
# 6. Sequence Creation
# -----------------------------------------------------------------------------
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

# 6.1 Create Sample Weights for Imbalance (0:1, 1:4)
# Since this is multi-label (n_stocks outputs in one layer), 
# we calculate a per-sample weight by averaging the weights of all labels in that sample.
def get_sample_weights(y):
    # y shape: (N, n_stocks, 3)
    # Based on provided distribution: Pos=21%, Neu=57%, Neg=21%
    # Inverse frequency weights: Neg=1/0.21, Neu=1/0.57, Pos=1/0.21
    # Normalized weights (Neu=1.0): Neg=2.71, Neu=1.0, Pos=2.71
    y_classes = np.argmax(y, axis=-1)
    
    weights_matrix = np.ones_like(y_classes, dtype=float)
    weights_matrix[y_classes == 0] = 2.71 # Negative
    weights_matrix[y_classes == 1] = 1.00 # Neutral
    weights_matrix[y_classes == 2] = 2.71 # Positive
    
    return weights_matrix

print("Calculating sample weights...")
train_weights = get_sample_weights(y_train_seq)
# We usually don't weight validation for "purity" of metrics, 
# but we can apply it to loss if we want it to be comparable.
val_weights = get_sample_weights(y_val_seq)

# -----------------------------------------------------------------------------
# 7. Model Building (Hard Parameter Sharing for Classification)
# -----------------------------------------------------------------------------
input_layer = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

# Shared LSTM Bottom
x = LSTM(128, return_sequences=True)(input_layer)
x = Dropout(0.2)(x) 
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.2)(x)

# Shared Representation
x = Dense(64, activation='relu')(x)

# Multi-class Output: n_stocks * 3 neurons, reshaped to (n_stocks, 3)
x = Dense(n_stocks * 3)(x)
x = Reshape((n_stocks, 3))(x)
output_layer = Softmax(axis=-1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Optimizer & Loss
# Using Categorical Crossentropy for multi-output classification
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# -----------------------------------------------------------------------------
# 8. Training & Callbacks
# -----------------------------------------------------------------------------
print("Starting Training (Classification)...")

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    mode='min'
)

# Model Checkpoint
checkpoint = ModelCheckpoint(
    'best_model_classification.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

history = model.fit(
    X_train_seq, y_train_seq,
    sample_weight=train_weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_seq, y_val_seq, val_weights),
    shuffle=True,
    callbacks=[early_stopping, checkpoint],  # Added Callbacks
    verbose=1
)

# Load the best model specifically (in case restore_best_weights isn't used or for explicit safety)
if os.path.exists('best_model_classification.h5'):
    print("Loading best model from checkpoint...")
    model = tf.keras.models.load_model('best_model_classification.h5')

# -----------------------------------------------------------------------------
# 9. Evaluation
# -----------------------------------------------------------------------------
print("Evaluating Classification Performance...")

y_prob = model.predict(X_test_seq)
# y_prob shape: (N, n_stocks, 3)
y_pred = np.argmax(y_prob, axis=-1)
y_true = np.argmax(y_test_seq, axis=-1)

# Flatten everything for global report
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()

print("\nGlobal Classification Report:")
print(classification_report(y_true_flat, y_pred_flat, target_names=['Negative', 'Neutral', 'Positive']))

print("\nGlobal Confusion Matrix:")
cm = confusion_matrix(y_true_flat, y_pred_flat)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Global Confusion Matrix (3-Class)')
plt.savefig('confusion_matrix.png')
print("Saved Confusion Matrix to: confusion_matrix.png")

# -----------------------------------------------------------------------------
# 10. Visualization
# -----------------------------------------------------------------------------
# Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Binary Crossentropy Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['pr_auc'], label='Train PR-AUC')
plt.plot(history.history['val_pr_auc'], label='Val PR-AUC')
plt.title('Precision-Recall AUC')
plt.legend()
plt.tight_layout()
plt.savefig('classification_history.png')

# Confidence Distribution for a sample stock (Neutral class)
stock_idx = 0
plt.figure(figsize=(10, 6))
sns.histplot(y_prob[:, stock_idx, 1], bins=50, kde=True, color='gray', label='Neutral Prob')
sns.histplot(y_prob[:, stock_idx, 2], bins=50, kde=True, color='green', label='Positive Prob')
sns.histplot(y_prob[:, stock_idx, 0], bins=50, kde=True, color='red', label='Negative Prob')
plt.title(f'Prediction Confidence Distribution: {label_cols[stock_idx]}')
plt.xlabel('Probability')
plt.legend()
plt.savefig('confidence_distribution.png')

print("Saved plots: classification_history.png, confidence_distribution.png")
