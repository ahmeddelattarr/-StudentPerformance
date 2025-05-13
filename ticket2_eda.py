import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

df = pd.read_csv("student-mat.csv", sep=';')

print("Preview of the dataset:")
print(df.head())

print("\nSummary statistics for numerical features:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())    

correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation_matrix)

plt.figure(figsize=(15,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

features_to_plot = ['G1', 'G2', 'studytime', 'failures', 'absences']

for feature in features_to_plot:
    plt.figure()
    sns.scatterplot(data=df, x=feature, y='G3')
    plt.title(f'Relationship between {feature} and Final Grade (G3)')
    plt.xlabel(feature)
    plt.ylabel('G3')
    plt.tight_layout()
    plt.show()

plt.figure()
sns.histplot(df['G3'], kde=True, bins=15)
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

categorical_features = ['sex', 'schoolsup', 'famsup', 'higher', 'internet']

for cat in categorical_features:
    plt.figure()
    sns.boxplot(data=df, x=cat, y='G3')
    plt.title(f"G3 Distribution by {cat}")
    plt.tight_layout()
    plt.show()