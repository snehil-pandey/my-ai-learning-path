# Linear Regression - Summary Report

This file summarizes the Linear Regression learning path so far, including basic calculations and actual examples.  
All generated plots are stored in the `images/` folder.

---

## 1. Basic Calculations

Linear Regression fits a straight line to predict output \($y$\) from input \($X$\).  

**Key formulas:**

- Predicted value:  
  $y_hat = m * X + c$  

- Residuals (error):  
  $residual = y - y_hat$  

- Mean Squared Error (MSE):  
  $MSE = mean(residual^2)$  

- Slope (m):  
  $m = sum((X - mean(X)) * (y - mean(y))) / sum((X - mean(X))^2)$  

- Intercept (c):  
  $c = mean(y) - m * mean(X)$  

**Explanation:**
- Slope = how much y changes per unit X  
- Intercept = baseline value when X = 0  
- MSE = how well the line fits the data  

All of these calculations were implemented manually in `1_basic_calculations/linear_regression_basics.py`.

**Sample Dataset:**  
$X_i$: `[1, 2, 3, 4, 5]`  
$y_i$: `[2, 4, 5, 4, 5]`

**Results**:
- Slope: `m = 0.60`
- Inctercept: `c = 2.20`
- Predicted $\hat{y}_i$ for respective $X_i$: `[2.8000000000000003, 3.4000000000000004, 4.0, 4.6, 5.2]`
- Residuals: `[-0.8000000000000003, 0.5999999999999996, 1.0, -0.5999999999999996, -0.20000000000000018]`
- Mean Squared Error: `MSE = 0.48`

---

## 2. Actual Examples

We applied Linear Regression to **two examples** using Python and scientific libraries:

### 2.1 Lottery Prediction (Toy Example)

- Dataset: Previous lottery draw numbers (toy example)  
- Purpose: Demonstrate linear trend even on random-like small data  

**Results:**  

- Slope: `m = 0.21`   
- Intercept: `c = 7.47`  
- MSE: `4.69`  

**Visualizations:**  
![Lottery Regression](images/lottery_prediction.png)  
*(Scatter plot + regression line)*

---

### 2.2 Temperature Prediction

- Dataset: Average daily temperatures over 10 days  
- Purpose: Show linear trend in real-world-like data  

**Results:**  

- Slope: `m = 1.04`  
- Intercept: `c = 29.67`  
- MSE: `0.28`  

**Visualizations:**  
![Temperature Regression](images/temperature_prediction.png)  
*(Scatter plot + regression line)*

**Residuals Plot:**  
![Temperature Residuals](images/temperature_residuals.png)  
*(Shows difference between actual and predicted values)*

---

## 3. Key Takeaways

- Linear Regression is **interpretable and foundational**  
- Can be done **manually** or with libraries for visualization  
- Residuals help understand **fit quality**  
- MSE is a key measure of **prediction accuracy**  
- These examples build the intuition needed for **advanced techniques** like gradient descent and multiple features  
