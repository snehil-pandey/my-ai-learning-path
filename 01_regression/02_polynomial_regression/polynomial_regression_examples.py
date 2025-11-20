"""
Polynomial Regression - Real Life Style Examples

This script demonstrates PRACTICAL examples of Polynomial Regression using the
core math functions defined in polynomial_regression_basics.py:

Includes:
- Example 1: Lottery Number Trend (Toy Example)
- Example 2: Temperature Forecasting
- Polynomial fitting (degree = 2 or 3)
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Import Required Functions
# ------------------------------

from polynomial_regression_basics import (
    compute_coefficients,
    predict_many,
    mean_squared_error
)

# ================================================================
# 2. Example 1 – Lottery Prediction (Toy Trend Example)
# ================================================================

print("Lottery Prediction")

# X_lottery : draw number (1 to 10)
# y_lottery : hypothetical winning numbers
X_lottery = np.array([1,2,3,4,5,6,7,8,9,10])
y_lottery = np.array([5,11,8,7,10,6,12,9,11,7])

degree_lottery = 3  # polynomial curve (captures fluctuations)

# --- Step 1: Compute polynomial coefficients ---
coeff_lottery = compute_coefficients(X_lottery.tolist(), y_lottery.tolist(), degree_lottery)

print(f"Polynomial Coefficients (lottery): {coeff_lottery}")

# --- Step 2: Prediction ---
y_pred_lottery = predict_many(X_lottery.tolist(), coeff_lottery)

# --- Step 3: Error ---
mse_lottery = mean_squared_error(y_lottery.tolist(), y_pred_lottery)
print(f"MSE (Lottery): {mse_lottery:.2f}")

# --- Step 4: Plot ---
plt.figure(figsize=(8,5))
plt.scatter(X_lottery, y_lottery, label="Actual Numbers")
plt.plot(X_lottery, y_pred_lottery, label="Polynomial Fit", linewidth=2)
plt.title("Lottery Trend Prediction (Toy Example)")
plt.xlabel("Draw Number")
plt.ylabel("Winning Number")
plt.grid(True)
plt.legend()
plt.savefig("./images/lottery_polynomial_fit.png")


# ================================================================
# 3. Example 2 – Temperature Prediction
# ================================================================

print("Temperature Prediction")

# X_temp : day number (1–10)
# y_temp : average daily temperature (°C)
X_temp = np.array([1,2,3,4,5,6,7,8,9,10])
y_temp = np.array([30,32,34,33,35,36,37,38,39,40])

degree_temp = 3  # gentle curve usually fits temperature well

# --- Step 1: Compute coefficients ---
coeff_temp = compute_coefficients(X_temp.tolist(), y_temp.tolist(), degree_temp)
print(f"Polynomial Coefficients (temperature): {coeff_temp}")

# --- Step 2: Predict ---
y_pred_temp = predict_many(X_temp.tolist(), coeff_temp)

# --- Step 3: Error ---
mse_temp = mean_squared_error(y_temp.tolist(), y_pred_temp)
print(f"MSE (Temperature): {mse_temp:.2f}")

# --- Step 4: Plot Actual vs Predicted ---
plt.figure(figsize=(8,5))
plt.scatter(X_temp, y_temp, color="green", label="Actual Temp")
plt.plot(X_temp, y_pred_temp, color="orange", label="Polynomial Fit", linewidth=2)
plt.title("Temperature Prediction Example")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.savefig("./images/temperature_polynomial_fit.png")


# --- Step 5: Residual Plot ---
residuals_temp = (y_temp - np.array(y_pred_temp))

plt.figure(figsize=(8,5))
plt.scatter(X_temp, residuals_temp, color="purple")
plt.hlines(0, xmin=min(X_temp), xmax=max(X_temp), colors="black", linestyles="dashed")
plt.title("Residuals - Temperature Prediction")
plt.xlabel("Day")
plt.ylabel("Residual")
plt.grid(True)
plt.savefig("./images/temperature_polynomial_residuals.png")
