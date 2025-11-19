"""
Linear Regression - Basic Calculations

This script demonstrates the fundamental calculations of Linear Regression:
- Calculating slope and intercept
- Predicting values
- Calculating residuals
- Computing Mean Squared Error (MSE)

All calculations are implemented using simple Python functions.
"""

# ------------------------------
# 1. Define Functions
# ------------------------------

def mean(values):
    """Calculate mean of a list of numbers."""
    return sum(values) / len(values)

def calculate_slope(X, y):
    """Calculate slope (m) for linear regression."""
    mean_x = mean(X)
    mean_y = mean(y)
    numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(len(X)))
    denominator = sum((X[i] - mean_x) ** 2 for i in range(len(X)))
    return numerator / denominator

def calculate_intercept(X, y, m):
    """Calculate intercept (c) for linear regression."""
    return mean(y) - m * mean(X)

def predict(X, m, c):
    """Predict y values based on slope and intercept."""
    return [m * X[i] + c for i in range(len(X))]

def calculate_residuals(y, y_pred):
    """Calculate residuals (errors) between actual and predicted values."""
    return [y[i] - y_pred[i] for i in range(len(y))]

def mean_squared_error(y, y_pred):
    """Calculate Mean Squared Error (MSE)."""
    residuals = calculate_residuals(y, y_pred)
    mse = sum(r**2 for r in residuals) / len(y)
    return mse

# ------------------------------
# 2. Example Dataset
# ------------------------------

X = [1, 2, 3, 4, 5]      # Independent variable
y = [2, 4, 5, 4, 5]      # Dependent variable

# ------------------------------
# 3. Calculations
# ------------------------------

# 3.1 Slope and Intercept
m = calculate_slope(X, y)
c = calculate_intercept(X, y, m)
print(f"Slope (m): {m:.2f}")
print(f"Intercept (c): {c:.2f}")

# 3.2 Predicted values
y_pred = predict(X, m, c)
print(f"Predicted y values: {y_pred}")

# 3.3 Residuals
residuals = calculate_residuals(y, y_pred)
print(f"Residuals: {residuals}")

# 3.4 Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")