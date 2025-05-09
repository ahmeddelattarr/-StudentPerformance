
# 📚 Student Performance Analysis - Full Project Documentation

## 📝 Objective

Analyze student data and build regression models to predict students' final grades (G3).

---

## 🗂️ Step 1: Import Libraries & Load Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
```

### 📖 Theory
- **pandas**: Data manipulation.
- **numpy**: Mathematical operations.
- **matplotlib/seaborn**: Data visualization.
- **sklearn**: Machine Learning models and utilities.

---

## 🗂️ Step 2: Load the Dataset

```python
df = pd.read_csv(r"Data\student-mat.csv", sep=";")
print(df.head())
```

### 📖 Theory
- Data read from CSV. Separator is `;` (common in European CSV formats).
- Initial look at the first 5 rows.

---

## 🧹 Step 3: Data Inspection

```python
print(df.isnull().sum())
print(df.dtypes)
```

### 📖 Theory
- **Missing Values**: Detect null or missing values.
- **Data Types**: Confirm the type of each column (numerical, categorical).

---

## 🧹 Step 4: Handle Missing Values

```python
df = df.dropna()
```

### 📖 Theory
- **dropna()**: Removes any row with missing data.
- **Note**: In real-world scenarios, sometimes filling missing data (imputation) is better than dropping rows.

---

## 🏷️ Step 5: Convert Categorical Variables (Encoding)

```python
df = pd.get_dummies(df, drop_first=True)
```

### 📖 Theory
- **One-Hot Encoding**: Converts categorical variables into binary columns (0/1).
- **drop_first=True**: Avoids the **Dummy Variable Trap** (redundancy).

---

## 🔧 Step 6: Feature Scaling

```python
scaler = StandardScaler()
num_cols = ['age', 'absences', 'G1', 'G2']
df[num_cols] = scaler.fit_transform(df[num_cols])
```

### 📖 Theory
- **StandardScaler**: Scales features to mean = 0 and std deviation = 1.
- **Why?**: Models like SVM and Gradient Boosting are sensitive to feature scales.

---

## 🔎 Step 7: Train-Test Split

```python
X = df.drop("G3", axis=1)
y = df["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 📖 Theory
- **X**: Features (inputs).
- **y**: Target variable (final grade - G3).
- **train_test_split**: Splits data into 80% training and 20% testing.

---

## 🤖 Step 8: Model Building

### 1️⃣ Linear Regression

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

**Theory**:
Linear Regression tries to fit the best possible straight line (in multi-dimensional space if more than one feature) that minimizes the squared difference between the actual and predicted values (**Ordinary Least Squares** method). It's best for simple, linear relationships.

### 2️⃣ Decision Tree Regressor

```python
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

**Theory**:
A Decision Tree splits the data into regions by asking "yes/no" questions at each node. Each split aims to reduce the prediction error (often using **Mean Squared Error** in regression trees). Trees can capture non-linear patterns but are prone to **overfitting** if not controlled.

### 3️⃣ Support Vector Regressor (SVR)

```python
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
```

**Theory**:
SVR tries to fit the best hyperplane that predicts data points while allowing some error within a specified margin (called **epsilon**). It focuses on the data points closest to the hyperplane (**support vectors**) and is useful for datasets where the relationship between features and target is complex and non-linear.

### 4️⃣ Gradient Boosting Regressor

```python
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
```

**Theory**:
Gradient Boosting builds a series of small trees sequentially, where each new tree tries to correct the errors of the previous trees. It minimizes a loss function (like MSE) using **gradient descent**. This ensemble approach typically leads to high prediction accuracy, especially in structured/tabular data.

---

## 🧮 Step 9: Model Evaluation

```python
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

metrics_lr = evaluate_model(y_test, y_pred_lr)
metrics_dt = evaluate_model(y_test, y_pred_dt)
metrics_svr = evaluate_model(y_test, y_pred_svr)
metrics_gbr = evaluate_model(y_test, y_pred_gbr)
```

### 📖 Theory
- **MSE (Mean Squared Error)**: Penalizes large errors.
- **MAE (Mean Absolute Error)**: Average of absolute errors.
- **R² Score**: Proportion of variance explained by the model (closer to 1 is better).

---

## 🏆 Step 10: Model Comparison

Compare the metrics (MSE, MAE, R²) across all models to choose the best performer.

**Usually:**
- Lower **MSE** & **MAE** = Better.
- Higher **R²** = Better.

---

## 📝 Final Notes

- **Linear Regression** → Simple baseline, good for linear relationships.
- **Decision Tree** → Captures non-linear patterns, easy to interpret, but can overfit.
- **SVR** → Effective for complex, non-linear data but slower on large datasets.
- **Gradient Boosting** → Often delivers the best accuracy for structured/tabular data.

---

## 📌 Summary Diagram

**Flow**:

```text
Data → Cleaning → Encoding → Scaling → Split → Model Training → Evaluation → Selection
```

---

## ✅ Ready for Exam

This document fully covers:
- Data Preprocessing concepts.
- Theory behind regression models.
- Detailed explanation of how each model works.
- Evaluation metrics.
