# Classification – Collective Notes
---
## 1. What is Classification?

Classification is a supervised ML task where the model learns to assign an input `X` into a **discrete class/category**.

Examples:
- Spam / Not Spam  
- Fraud / Not Fraud  
- Cat / Dog  
- Pass / Fail  
- Disease / No Disease  

Goal:

$f(X) \rightarrow \text{Class label}$

Unlike regression (continuous output), classification predicts **categorical output**.

---

## 2. Why Do We Use Classification?

- Detect spam/fraud  
- Medical diagnosis  
- Risk assessment  
- Sentiment analysis  
- Image/object recognition  
- Customer churn prediction  

Classification is fundamental in almost every real-world ML system.

---

## 3. Types of Classification

### 3.1 Binary Classification
Only **two classes**  
Examples:  
- Approve Loan / Reject Loan  
- Spam / Not Spam  

---

### 3.2 Multi-Class Classification
More than two classes  
Examples:  
- Classifying images into 10 digits (0–9)  
- Categorizing news articles  

---

### 3.3 Multi-Label Classification
Each input can have **multiple classes simultaneously**.  
Examples:  
- A photo tagged with: `["outdoor", "person", "night"]`  
- Text categories like: `["politics", "crime"]`

---

# 4. Major Classification Algorithms

### 4.1 Logistic Regression
Despite the name, it is a **classification algorithm**.

Predicts **probability** using sigmoid:

$P(y=1|X) = \sigma(w^Tx)$

Decision rule:

$$
\hat{y} = 
\begin{cases}
1, & P \ge 0.5\\
0, & P < 0.5
\end{cases}
$$

Used for:
- Simple, linear separable data  
- Quick baseline models  

---

### 4.2 K-Nearest Neighbors (KNN)
Instance-based classifier.

Steps:
1. Find the **K nearest points**  
2. Take **majority class** among them  

Distance metric:  
- Euclidean  
- Manhattan  

Best for:
- Small datasets  
- Non-linear boundaries  

---

### 4.3 Support Vector Machine (SVM)
Finds the **optimal separating hyperplane** with **maximum margin**.

Decision rule uses:

$w^Tx + b = 0$

For non-linear data, uses **Kernels** (RBF, Polynomial).

Best for:
- High-dimensional data  
- Small datasets  
- Clear boundaries  

---

### 4.4 Decision Trees
Tree structure:  
- Nodes &rarr; questions  
- Leaves &rarr; class labels  

Advantages:
- Easily interpretable  
- Handles non-linear relationships  

---

### 4.5 Random Forest
Ensemble of Decision Trees.  
Uses:
- Bootstrap samples  
- Feature randomness  
- Majority voting  

Advantages:
- High accuracy  
- Handles noise + outliers  

---

### 4.6 Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)
Builds trees **sequentially** to fix errors of previous trees.

Strongest algorithms for:
- Tabular data  
- Kaggle competitions  
- Business ML systems  

---

### 4.7 Naive Bayes
Probabilistic classifier based on **Bayes theorem**:

$$
P(class|data) = \frac{P(data|class)\, P(class)}{P(data)}
$$

Assumes feature independence (hence “naive”).

Best for:
- Text classification  
- Spam filtering  

---

### 4.8 Neural Network Classifiers
Deep learning models that learn complex, non-linear patterns.

Used for:
- Image classification (CNNs)  
- Text classification (RNNs/Transformers)  
- Voice recognition  

---

## 5. Core Mathematics

### 5.1 Logistic Regression Math

#### Sigmoid Function:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

#### Loss Function: Binary Cross Entropy

$L = -[y\ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$

Optimized using **Gradient Descent**.

---

### 5.2 SVM Optimization

Objective:

$\min \frac{1}{2}\|w\|^2$
Subject to correct classification and margin constraints.

Kernel Trick:

$K(x_i,x_j) = \phi(x_i)\cdot\phi(x_j)$

---

### 5.3 Decision Tree Math (Gini / Entropy)

#### Entropy:

$H = -\sum p_i \log_2 p_i$

#### Gini Index:

$G = 1 - \sum p_i^2$

Split chosen to **maximize Information Gain**.

---

### 5.4 Bayes Theorem

$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

Foundation of Naive Bayes.

---

## 6. Classification Workflow

1. **Collect Data**  
2. **Preprocess** (clean, encode, scale)  
3. **Split train/test**  
4. **Choose classifier**  
5. **Train model**  
6. **Evaluate**  
7. **Tune hyperparameters**  
8. **Deploy**  

---

## 7. Evaluation Metrics

### 7.1 Confusion Matrix

|                | Pred 0 | Pred 1 |
|----------------|--------|--------|
| **Actual 0**   | TN     | FP     |
| **Actual 1**   | FN     | TP     |

---

### 7.2 Key Metrics

#### Accuracy
$\frac{TP + TN}{Total}$

#### Precision
$\frac{TP}{TP + FP}$

#### Recall
$\frac{TP}{TP + FN}$

#### F1 Score
$2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

#### ROC-AUC
Measures ranking quality of predicted probabilities.

---

## 8. When to Use Which Classifier?

| Scenario | Best Choice |
|----------|-------------|
| Linearly separable data | Logistic / Linear SVM |
| Small dataset | SVM / KNN |
| High dimensional (text) | Naive Bayes / Linear SVM |
| Large tabular data | Random Forest / XGBoost |
| Non-linear patterns | Trees / Kernel SVM |
| Images, audio, NLP | Neural Networks |
| Need interpretability | Logistic / Decision Tree |

---

## 9. Intuition Behind Classification

- Classification draws **boundaries** between categories.  
- Logistic regression &rarr; probability-based decision boundary.  
- KNN &rarr; “vote of nearest neighbors".  
- SVM &rarr; “max margin separation".  
- Trees &rarr; split space logically.  
- Ensembles &rarr; many weak learners &rarr; strong model.  
- Neural nets &rarr; layers of transformations &rarr; pattern recognition.  

In short:
**Classification = Deciding which bucket an input belongs to.**

---

## 10. Summary

- Classification predicts **categories**, not continuous values.  
- Multiple algorithms exist, each suited to different data types.  
- Evaluation uses precision, recall, F1, ROC-AUC, etc.  
- Choosing the right classifier depends on data size, shape, complexity, and business requirement.

This document serves as a **complete overview of classification techniques** with clean theory + math + intuition.

---
