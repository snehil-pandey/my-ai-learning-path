# Regression – Collective Notes

## 1. What is Regression?

Regression is a statistical + machine learning technique used to model the **relationship between an independent variable (X)** and a **dependent variable (y)**.

Goal:
- Understand how `y` changes with respect to `X`
- Build a function `f(X)` that predicts `y`
- Capture patterns, trends, and relationships in data

General form:

$y = f(X) + \epsilon$

Where  
- $f(X)$ = model  
- \($\epsilon$\) = error/noise

---

## 2. Why Do We Use Regression?

- Predict future values  
- Identify trends  
- Understand influence of factors  
- Fill missing data  
- Analyze relationships between variables  
- Build ML models for business, engineering, finance, etc.

---
## 3. Types of Regression

### 3.1 Linear Regression
Models a **straight line** relationship:

$y = b_0 + b_1x$

Uses **Least Squares Method** to minimize:

$\sum (y - \hat{y})^2$

Best For:
- Linear patterns  
- Strong causal/monotonic relationships  

---

### 3.2 Multiple Linear Regression
More than one input variable:

$y = b_0 + b_1x_1 + b_2x_2 + \dots + b_kx_k$

Captures influence of multiple factors.  
Common in business, analytics, and engineering.

---

### 3.3 Polynomial Regression
Used when the relationship is **curved**.

$y = b_0 + b_1x + b_2x^2 + \dots + b_dx^d$

We convert `X` into polynomial features and solve using:

$A X = B \quad\Rightarrow\quad X = A^{-1}B$

Useful when:
- Data has bends or non-linear patterns  
- You want smooth curves  

---

### 3.4 Regularized Regression
Adds penalty terms to avoid overfitting.

#### Ridge Regression (L2)
$$
\text{Loss} = \sum (y-\hat{y})^2 + \lambda\sum b_i^2
$$

#### Lasso Regression (L1)
$$
\text{Loss} = \sum (y-\hat{y})^2 + \lambda\sum |b_i|
$$

#### Elastic Net = L1 + L2 combined

Used when:
- Too many features  
- Multicollinearity
- Preventing overfitting

---

### 3.5 Logistic Regression (Classification)
Despite the name, it predicts **probability**, not regression values.

$$
P(y=1|X) = \sigma(b_0 + b_1x)
$$

Used for:
- Spam/not spam  
- Disease/no disease  
- Fraud/not fraud  

---

### 3.6 Other Regressors (Advanced ML)
- **Decision Tree Regression**
- **Random Forest Regression**
- **Gradient Boosted Regression**
- **Support Vector Regression (SVR)**
- **Neural Network Regression**

Used when:
- Data is complex  
- Patterns are highly non-linear  
- Traditional methods fail  

---
## 4. Core Mathematics

### 4.1 Linear Regression Normal Equation
$X = (A^TA)^{-1}A^Ty$

Where:
- `A` → design matrix  
- `X` → coefficients  
- `y` → outputs  

---

### 4.2 Polynomial Regression Matrix (degree d)

A =
$\begin{bmatrix}
n & \sum x & \sum x^2 & \dots & \sum x^d \\
\sum x & \sum x^2 & \sum x^3 & \dots & \sum x^{d+1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\sum x^d & \sum x^{d+1} & \dots & \dots & \sum x^{2d}
\end{bmatrix}$

B =
$$
\begin{bmatrix}
\sum y \\
\sum xy \\
\vdots \\
\sum x^d y
\end{bmatrix}
$$

Solve:

$X = A^{-1}B$

---

## 5. Workflow of Any Regression

1. **Collect Data**  
2. **Clean Data**  
3. **Split Data (train/test)**  
4. **Choose model** (Linear, Poly, Ridge…)  
5. **Train model** (solve AX=B or gradient descent)  
6. **Evaluate**  
7. **Tune hyperparameters**  
8. **Deploy & predict**

---

## 6. Evaluation Metrics

### Regression Metrics
| Metric | Meaning |
|-------|---------|
| **MAE** | Average absolute error |
| **MSE** | Penalizes large errors |
| **RMSE** | Square-root of MSE |
| **R² Score** | Proportion of variance explained |

High-quality model:
- High R²  
- Low MSE/RMSE  

---

## 7. When to Use Which Regression?

| Situation | Best Regression |
|-----------|-----------------|
| Straight-line trend | Linear |
| Curve but smooth | Polynomial |
| Many correlated features | Ridge |
| Need feature selection | Lasso |
| Complex patterns | Random Forest / GBDT |
| Probability output | Logistic |
| Outliers present | Robust Regression |

---

## 8. Intuition Behind Regression

- Regression draws a **line/curve that represents the trend** of your data.
- The best model is the one that **minimizes total error**.
- Higher-degree polynomials fit more curves but risk **overfitting**.
- Regularization controls complexity by **punishing large coefficients**.
- Logistic regression is basically a linear model passed through a **sigmoid** to make probabilities.

---

## 9. Summary

- Regression predicts relationships between variables.  
- Linear = straight line.  
- Polynomial = curved line.  
- Regularized = safer, stable, less overfit.  
- Logistic = classification.  
- Advanced methods handle messy real-world data.

This document acts as a **final combined reference** for all regression concepts studied so far.

---
