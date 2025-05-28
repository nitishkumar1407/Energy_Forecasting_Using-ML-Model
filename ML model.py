import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
import joblib

# Load Dataset
data = pd.read_csv(r"C:\Users\harip\OneDrive\Desktop\Energy_Forecasting_Using-ML-Model\mp_citywise_energy_temp_daily_2019_2024.csv", encoding='latin1')

# Check required columns
required_cols = ['City', 'Year', 'Month', 'Date', 'Temperature (°C)', 'Total_Consumption_kWh']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Extract date-based features
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['dayofweek'] = data['Date'].dt.dayofweek
data['dayofyear'] = data['Date'].dt.dayofyear

# Create dummy variables for City (categorical)
X = pd.get_dummies(data[['month', 'day', 'dayofweek', 'dayofyear', 'City']], drop_first=True)

# Add numeric feature
X['Temperature (°C)'] = data['Temperature (°C)']

# Define target variable
y = data['Total_Consumption_kWh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Meta-learner with scaling
meta_model = make_pipeline(StandardScaler(), LinearRegression())

# Stacking Regressor
stack_model = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=meta_model
)

# Train the model
stack_model.fit(X_train, y_train)

# Predict on test data
y_pred = stack_model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Evaluation Results:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# Save the model
joblib.dump(stack_model, "model.pkl")
print("Model training complete and saved as 'model.pkl'")