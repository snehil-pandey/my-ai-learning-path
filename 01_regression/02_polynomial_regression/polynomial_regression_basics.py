"""
Polynomial Regression - Basic Calculations (Summation Formula Method)

This script demonstrates the fundamental calculations of Polynomial Regression
using ONLY the classical summation-based normal equations.

Concepts implemented:
- Polynomial feature powers (x^0 ... x^d)
- Constructing matrix A where A[i][j] = Σ x^(i+j)
- Constructing vector B where B[i] = Σ (x^i * y)
- Matrix inversion using Gauss–Jordan elimination (from scratch)
- Solving for coefficients:  X = A^-1 * B
- Predicting output values using the polynomial model
- Computing residuals and Mean Squared Error (MSE)

This file is purely educational — NO numpy is used.
"""

# ------------------------------
# 1. Define Functions
# ------------------------------

def poly_features(x, degree):
    """
    Returns: [x^0, x^1, x^2, ..., x^degree]
    Example:
        x = 3, degree = 3 → [1, 3, 9, 27]
    """
    return [x ** i for i in range(degree + 1)]


def build_matrix_A(x_values, degree):
    """
    Construct the coefficient matrix A for polynomial regression.

    A[i][j] = Σ x^(i+j)

    Shape: (degree+1) × (degree+1)
    """
    n = degree + 1
    A = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            power = i + j
            A[i][j] = sum(x ** power for x in x_values)

    return A


def build_matrix_B(x_values, y_values, degree):
    """
    Construct vector B where:
        B[i] = Σ (x^i * y)
    """
    n = degree + 1
    B = [0 for _ in range(n)]

    for i in range(n):
        B[i] = sum((x_values[k] ** i) * y_values[k] for k in range(len(x_values)))

    return B


def invert_matrix(matrix):
    """
    Compute inverse of a square matrix using Gauss–Jordan elimination.

    This version is simple and educational — not optimized for speed.
    """
    n = len(matrix)

    # Identity matrix
    identity = [[float(i == j) for j in range(n)] for i in range(n)]

    # Copy input matrix
    mat = [row[:] for row in matrix]

    for i in range(n):

        # Step 1: If pivot is zero, try swapping
        if mat[i][i] == 0:
            for j in range(i + 1, n):
                if mat[j][i] != 0:
                    mat[i], mat[j] = mat[j], mat[i]
                    identity[i], identity[j] = identity[j], identity[i]
                    break

        pivot = mat[i][i]

        # Normalize the pivot row
        for col in range(n):
            mat[i][col] /= pivot
            identity[i][col] /= pivot

        # Eliminate above & below pivot
        for row in range(n):
            if row != i:
                factor = mat[row][i]
                for col in range(n):
                    mat[row][col] -= factor * mat[i][col]
                    identity[row][col] -= factor * identity[i][col]

    return identity


def multiply_matrix_vector(matrix, vector):
    """
    Multiply matrix (m×n) with vector (n).
    """
    result = [0 for _ in range(len(matrix))]

    for i in range(len(matrix)):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(len(vector)))

    return result


def compute_coefficients(x_values, y_values, degree):
    """
    Full regression pipeline:

    1. Build Matrix A (Σ x^(i+j))
    2. Build Vector B (Σ x^i * y)
    3. Compute A^-1 using Gauss–Jordan
    4. Compute X = A^-1 * B

    Returns: list of coefficients [a0, a1, ..., ad]
    """
    A = build_matrix_A(x_values, degree)
    B = build_matrix_B(x_values, y_values, degree)
    A_inv = invert_matrix(A)
    X = multiply_matrix_vector(A_inv, B)
    return X


def predict(x, coefficients):
    """
    Predict y = a0 + a1*x + a2*x^2 + ... + ad*x^d
    """
    return sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))


def predict_many(x_values, coefficients):
    """
    Predict multiple outputs for a list of x values.
    """
    return [predict(x, coefficients) for x in x_values]


def calculate_residuals(y, y_pred):
    """
    Residuals = actual - predicted
    """
    return [y[i] - y_pred[i] for i in range(len(y))]


def mean_squared_error(y, y_pred):
    """
    Compute MSE = Σ (error^2) / n
    """
    residuals = calculate_residuals(y, y_pred)
    return sum(r ** 2 for r in residuals) / len(y)


# ------------------------------
# 2. Example Dataset
# ------------------------------

X = [1, 2, 3, 4, 5]      # Independent variable values
y = [2, 4, 5, 4, 5]      # Observed target values
degree = 3               # Degree of polynomial model


# ------------------------------
# 3. Calculations
# ------------------------------

# 3.1 Compute polynomial coefficients
coefficients = compute_coefficients(X, y, degree)
print(f"Polynomial Coefficients: {coefficients}")

# 3.2 Compute predicted y values
y_pred = predict_many(X, coefficients)
print(f"Predicted y values: {y_pred}")

# 3.3 Compute residuals
residuals = calculate_residuals(y, y_pred)
print(f"Residuals: {residuals}")

# 3.4 Compute mean squared error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
