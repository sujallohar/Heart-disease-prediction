import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import recall_score, f1_score, precision_score

#reading dataset
ds=pd.read_csv(r"/content/heart.csv",header="infer")
print(ds.head(10))
print(ds.info())
print(ds.shape)
class_label = 'HeartDisease'
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

#encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    ds[col] = label_encoder.fit_transform(ds[col])

#handling missing values
mf_exercise_angina = ds['ExerciseAngina'].mode()[0]
ds['ExerciseAngina'].fillna(mf_exercise_angina, inplace=True)

mean_resting_bp = ds['RestingBP'].mean()
ds['RestingBP'].fillna(mean_resting_bp, inplace=True)

print(ds.info())

#statistical analysis
statistical_analysis = ds.describe()
print("\nStatistical Analysis:\n", statistical_analysis)

variance = ds[numerical_columns].var()
std_deviation = ds[numerical_columns].std()
print("\nVariance:\n", variance)
print("Standard Deviation:\n", std_deviation)

median = ds[numerical_columns].median()
print("\nMedian:\n", median)

mode = ds[numerical_columns].mode().iloc[0]
print("\nMode:\n", mode)

#unique values and unique value counts
for column in ds.columns:
    unique_values = ds[column].unique()
    value_counts = ds[column].value_counts()

    print(f"Column: {column}")
    print("Unique Values:")
    print(unique_values)
    print("\nValue Counts:")
    print(value_counts)
    print("\n")

#plotting

#histogram
plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    plt.hist(ds[column], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(column)
plt.tight_layout()
plt.show()

#bar graph
plt.figure(figsize=(12, 8))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(2, 4, i)
    ds[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(column)
plt.tight_layout()
plt.show()

#scatter plot
plt.figure(figsize=(12, 8))
for i, col1 in enumerate(numerical_columns):
    for j, col2 in enumerate(numerical_columns):
        if i != j:
            if (i * len(numerical_columns) + j + 1) <= 16:
                plt.subplot(4, 4, i * len(numerical_columns) + j + 1)
                plt.scatter(ds[col1], ds[col2], alpha=0.5)
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.title(f"{col1} vs {col2}")

plt.tight_layout()
plt.show()

#training the model (KNN)
X = ds.drop(columns=['HeartDisease'])
y = ds['HeartDisease']

ts=float(input("enter test size(between 0 to 1)"))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, stratify=y)


k = int(input("Enter the number of nearest neighbors to be used, i.e. k: "))
model = KNeighborsClassifier(n_neighbors=k, weights='distance')
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, pred))

k_values = []
accuracies = []
recalls = []
f1_scores = []
precisions = []

k_range = range(1, k+1)

for k_val in k_range:
    model = KNeighborsClassifier(n_neighbors=k_val, weights='distance')

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)

    k_values.append(k_val)
    accuracies.append(accuracy)
    recalls.append(recall)
    f1_scores.append(f1)
    precisions.append(precision)

# Plotting the graphs
plt.figure(figsize=(12, 6))

#accuracy vs k
plt.figure(figsize=(10, 6))
plt.bar(k_values, accuracies, linestyle='-')
plt.title('Accuracy vs Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.show()

#recall vs k
plt.subplot(1, 3, 1)
plt.bar(k_values, recalls, linestyle='-')
plt.title('Recall vs Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Recall')
plt.grid(True)

#f1 score vs k
plt.subplot(1, 3, 2)
plt.bar(k_values, f1_scores, linestyle='-')
plt.title('F1 Score vs Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('F1 Score')
plt.grid(True)

#precision vs k
plt.subplot(1, 3, 3)
plt.bar(k_values, precisions, linestyle='-')
plt.title('Precision vs Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Precision')
plt.grid(True)

plt.tight_layout()
plt.show()

