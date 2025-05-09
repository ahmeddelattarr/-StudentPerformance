# 📚 Student Performance Analysis - Full Project Documentation

## 📝 Objective

Analyze student data and build regression models to predict students' final grades (**G3**) using various machine learning techniques. The goal is to identify the best-performing model based on evaluation metrics.

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
```

### 📖 Theory
- **pandas**: Data manipulation.
- **numpy**: Mathematical operations.
- **matplotlib/seaborn**: Data visualization.
- **sklearn**: Machine learning models and utilities.

---

## 🗂️ Step 2: Load the Dataset

```python
df = pd.read_csv(r"Data\student-mat.csv", sep=";")
print(df.head())
```

### 📖 Theory
- The dataset is loaded from a CSV file with `;` as the separator.
- The first few rows are displayed to understand the structure of the data.

---

## 🧹 Step 3: Data Preprocessing

### 1️⃣ Inspect the Data

```python
print(df.isnull().sum())
print(df.dtypes)
```

### 📖 Theory
- **Missing Values**: Check for null or missing values.
- **Data Types**: Confirm the type of each column (numerical or categorical).

---

### 2️⃣ Handle Missing Values

```python
df = df.dropna()
```

### 📖 Theory
- **dropna()**: Removes rows with missing values.
- **Alternative**: Impute missing values (e.g., mean, median) if dropping rows results in significant data loss.

---

### 3️⃣ Encode Categorical Variables

```python
df = pd.get_dummies(df, drop_first=True)
```

### 📖 Theory
- **One-Hot Encoding**: Converts categorical variables into binary columns (0/1).
- **drop_first=True**: Avoids redundancy by dropping one category.

---

## 🔎 Step 4: Exploratory Data Analysis (EDA)

### 1️⃣ Summary Statistics

```python
print(df.describe())
```

### 📖 Theory
- Provides an overview of numerical features, including mean, standard deviation, and range.
- Helps identify potential outliers and data distributions.

---

### 2️⃣ Correlation Analysis

```python
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
```

### 📖 Theory
- **Correlation Matrix**: Identifies relationships between features and the target variable (`G3`).
- **Why?**: Helps select features that are strongly correlated with `G3`.

---

### 3️⃣ Feature Relationships with Final Grade (G3)

```python
features_to_plot = ['G1', 'G2', 'studytime', 'failures', 'absences']
for feature in features_to_plot:
    sns.scatterplot(data=df, x=feature, y='G3')
    plt.title(f'Relationship between {feature} and Final Grade (G3)')
    plt.show()
```

### 📖 Theory
- **Scatterplots**: Visualize how individual features impact `G3`.
- **Key Insights**:
  - `G1` and `G2`: Strong positive relationships with `G3`.
  - `failures`: Negative relationship with `G3`.
  - `studytime` and `absences`: Weak relationships but still relevant.

---

## 🔧 Step 5: Feature Selection & Scaling

### 1️⃣ Feature Selection

```python
X = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y = df['G3']
```

### 📖 Theory
- Select features that are most relevant to predicting `G3`.

---

### 2️⃣ Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 📖 Theory
- Splits the data into training (80%) and testing (20%) sets to evaluate model performance.

---

### 3️⃣ Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 📖 Theory
- Scales features to ensure they are on the same scale, which is crucial for models like SVR and Gradient Boosting.

---

## 🤖 Step 6: Model Training

### Models Trained:
1. **Support Vector Regression (SVR)**
2. **Linear Regression**
3. **Gradient Boosting Regression**
4. **Decision Tree Regression (with Grid Search)**
Here’s a small tutorial for each model in the requested format:

---

### 1️⃣ Support Vector Regression (SVR)

```python
from sklearn.svm import SVR

# Initialize and train the SVR model
svr = SVR(kernel='linear')
svr.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_svr = svr.predict(X_test_scaled)
```

### 📖 Theory
- **Support Vector Regression (SVR)**: A regression model that fits a hyperplane to predict the target variable while allowing some error within a margin (epsilon).
- **Kernel**: The `linear` kernel is used here, but other kernels (e.g., `rbf`) can capture non-linear relationships.
- **Why Use It?**: Effective for datasets with complex relationships and works well with scaled features.

---

### 2️⃣ Linear Regression

```python
from sklearn.linear_model import LinearRegression

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lr = lr.predict(X_test_scaled)
```

### 📖 Theory
- **Linear Regression**: A simple regression model that fits a straight line (or hyperplane) to minimize the squared differences between actual and predicted values.
- **Why Use It?**: Easy to interpret and serves as a baseline model for linear relationships.

---

### 3️⃣ Gradient Boosting Regression

```python
from sklearn.ensemble import GradientBoostingRegressor

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_gbr = gbr.predict(X_test_scaled)
```

### 📖 Theory
- **Gradient Boosting Regression**: An ensemble model that builds a series of small decision trees sequentially, where each tree corrects the errors of the previous ones.
- **Why Use It?**: Highly accurate for structured/tabular data and robust to overfitting with proper tuning.

---

### 4️⃣ Decision Tree Regression with Grid Search

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize the Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)

# Perform Grid Search to find the best parameters
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)

# Get the best model and predict on the test set
best_decision_tree = grid_search.best_estimator_
y_pred_dt = best_decision_tree.predict(X_test)
```

### 📖 Theory
- **Decision Tree Regression**: Splits data into regions based on feature thresholds. Each split reduces prediction error.
- **Grid Search**: Tunes hyperparameters (e.g., tree depth, minimum samples per split) to find the best-performing model.
- **Why Use It?**: Captures non-linear patterns and is interpretable but requires tuning to avoid overfitting.

---

### Summary of Models:
- **SVR**: Best for datasets with complex relationships and scaled features.
- **Linear Regression**: A simple and interpretable baseline model.
- **Gradient Boosting**: Highly accurate and robust for structured data.
- **Decision Tree**: Captures non-linear patterns but requires careful tuning.

---

## 🧮 Step 7: Model Evaluation

### Evaluation Metrics:

```python
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2
```

- **RMSE**: Penalizes large errors more heavily.
- **MAE**: Measures average absolute error.
- **R²**: Indicates how well the model explains the variance in the data.

---

## 🏆 Step 8: Model Comparison

### Compare Models:

```python
models = {
    "SVR": results_svr,
    "Linear Regression": results_lr,
    "Gradient Boosting Regression": results_gbr,
    "Decision Tree Regression": results_dt
}

metric_to_plot = 'RMSE'  # or MAE or R²

# Extracting the values for the chosen metric from each model
metric_values = {model: metrics[metric_to_plot] for model, metrics in models.items()}

# Plotting the bar chart for the selected metric
plt.figure(figsize=(10, 6))
plt.barh(list(metric_values.keys()), list(metric_values.values()), color='skyblue')
plt.xlabel(f"{metric_to_plot} Value", fontsize=12)
plt.ylabel("Models", fontsize=12)
plt.title(f"Comparison of Models based on {metric_to_plot}", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

### 📖 Theory
- **RMSE**: Prioritized for selecting the best model as it penalizes large errors.
- **MAE**: Useful for understanding average error magnitude.
- **R²**: Supplementary metric to evaluate explanatory power.

---

## ✅ Best Model

Based on the evaluation metrics:
- **Gradient Boosting Regression** is the best model because it has:
  - The **lowest RMSE** (smallest average error magnitude).
  - The **highest R²** (best explanatory power).
  - Competitive MAE compared to other models.

---

This detailed documentation provides a comprehensive overview of the project, from data preprocessing to model evaluation and selection.