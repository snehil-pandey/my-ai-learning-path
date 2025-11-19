"""
Linear Regression - Actual Examples

This script demonstrates Linear Regression using small practical examples:
1. Lottery Prediction (toy random example)
2. Temperature Prediction (simple trend example)

We will visualize data, regression lines, residuals, and compute Mean Squared Error.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Define Functions (reuse from basics)
# ------------------------------

def mean(values):
    return np.mean(values)

def calculate_slope(X, y):
    mean_x = mean(X)
    mean_y = mean(y)
    numerator = np.sum((X - mean_x) * (y - mean_y))
    denominator = np.sum((X - mean_x)**2)
    return numerator / denominator

def calculate_intercept(X, y, m):
    return mean(y) - m * mean(X)

def predict(X, m, c):
    return m * X + c

def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred)**2)

# ------------------------------
# 2. Example 1: Lottery Numbers (Toy Example)
# ------------------------------

# Simulate previous lottery draws: Draw number vs winning number (toy data)
X_lottery = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_lottery = np.array([5, 11, 8, 7, 10, 6, 12, 9, 11, 7])  # hypothetical winning numbers

# Linear Regression calculations
m1 = calculate_slope(X_lottery, y_lottery)
c1 = calculate_intercept(X_lottery, y_lottery, m1)
y_pred_lottery = predict(X_lottery, m1, c1)
mse_lottery = mean_squared_error(y_lottery, y_pred_lottery)

print("Lottery Prediction Example")
print(f"Slope (m): {m1:.2f}, Intercept (c): {c1:.2f}")
print(f"MSE: {mse_lottery:.2f}")

# Plot
plt.figure(figsize=(8,5))
plt.scatter(X_lottery, y_lottery, color='blue', label='Actual lottery numbers')
plt.plot(X_lottery, y_pred_lottery, color='red', label='Predicted line')
plt.title('Lottery Numbers Prediction (Toy Example)')
plt.xlabel('Draw Number')
plt.ylabel('Winning Number')
plt.legend()
plt.grid(True)
plt.savefig('./images/lottery_prediction.png')

# ------------------------------
# 3. Example 2: Temperature Prediction
# ------------------------------

# Simulated average daily temperatures over 10 days
X_temp = np.array([1,2,3,4,5,6,7,8,9,10])  # day
y_temp = np.array([30, 32, 34, 33, 35, 36, 37, 38, 39, 40])  # temperature in °C

# Linear Regression calculations
m2 = calculate_slope(X_temp, y_temp)
c2 = calculate_intercept(X_temp, y_temp, m2)
y_pred_temp = predict(X_temp, m2, c2)
mse_temp = mean_squared_error(y_temp, y_pred_temp)

print("\nTemperature Prediction Example")
print(f"Slope (m): {m2:.2f}, Intercept (c): {c2:.2f}")
print(f"MSE: {mse_temp:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8,5))
plt.scatter(X_temp, y_temp, color='green', label='Actual temperature')
plt.plot(X_temp, y_pred_temp, color='orange', label='Regression line')
plt.title('Temperature Prediction Example')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.savefig('./images/temperature_prediction.png')

# Plot residuals for temperature example
residuals_temp = y_temp - y_pred_temp
plt.figure(figsize=(8,5))
plt.scatter(X_temp, residuals_temp, color='purple')
plt.hlines(y=0, xmin=min(X_temp), xmax=max(X_temp), colors='black', linestyles='dashed')
plt.title('Residuals - Temperature Prediction')
plt.xlabel('Day')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(True)
plt.savefig('./images/temperature_residuals.png')
