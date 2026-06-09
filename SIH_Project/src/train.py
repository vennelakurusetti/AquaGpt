import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
data_path = "../groundwater-level-collected-telemetric.csv"  # Adjust path if needed
df = pd.read_csv(data_path)
print("✅ Loaded successfully:", df.shape)

# Preprocessing
df = df.copy()

# Drop irrelevant/empty columns if present
for col in ['id', 'date_of_establishment', 'data_available_from']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Combine date + time into datetime (if present)
if {'date','time'}.issubset(df.columns):
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    df = df.drop(columns=['date','time'])
elif 'datetime' not in df.columns:
    df['datetime'] = pd.NaT

# Ensure target exists
if 'data_value' not in df.columns:
    raise ValueError("Expected 'data_value' column as target.")

# Drop rows without target
df = df.dropna(subset=['data_value']).copy()

# Fill numeric gaps
for col in ['discharge', 'well_depth']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical gaps with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Optional: time features (useful for ML, not the raw datetime)
if 'datetime' in df.columns:
    df['year']  = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day']   = df['datetime'].dt.day
    df['hour']  = df['datetime'].dt.hour

# Label-encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Normalize selected numeric columns to [0,1]
numeric_cols = [c for c in ['latitude','longitude','discharge','well_depth','data_value'] if c in df.columns]
scaler = MinMaxScaler()
if numeric_cols:
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save normalized dataset for reuse
df.to_csv("data/normalized_groundwater_data.csv", index=False)
print("✅ Saved normalized data")

# Prepare features and target
X = df.drop(columns=[c for c in ['data_value','datetime'] if c in df.columns]).copy()
y = df['data_value'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained")

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MSE: {mse:.6f}, R2: {r2:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_v1.pkl")
print("✅ Model saved to models/model_v1.pkl")