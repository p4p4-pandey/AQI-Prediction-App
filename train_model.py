import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Example training data
# Columns: PM2.5, PM10, NO2, SO2, CO, Temperature, Humidity
X = np.array([
    [50, 80, 40, 20, 1, 25, 50],
    [100, 150, 70, 40, 2, 30, 60],
    [30, 60, 20, 10, 0.5, 20, 40],
    [200, 300, 100, 80, 3, 35, 70],
    [80, 120, 50, 30, 1.5, 28, 55]
])

# AQI values (target)
y = np.array([100, 200, 60, 300, 150])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "LR_AQI_Prediction.joblib")

print("✅ Model trained and saved successfully!")
