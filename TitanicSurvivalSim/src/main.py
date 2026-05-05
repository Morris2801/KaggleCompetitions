"""
Titanic survival baseline workflow

General steps:
1. Load the training and test CSV files with pandas.
2. Inspect dataset shape, columns, missing values, and basic survival patterns.
3. Preprocess numeric and categorical features with a scikit-learn pipeline.
4. Train a Logistic Regression baseline model.
5. Evaluate the model on a validation split.
6. Retrain on all training data and write a submission CSV for the test set.

Tools used:
- pathlib: file paths that work no matter where the script is started from
- pandas: loading and manipulating CSV data
- matplotlib: quick exploration plots
- scikit-learn: preprocessing, model training, and evaluation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Build paths relative to this file so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# Load the Titanic datasets.
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Print the exploration results so they are visible in the terminal.
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain preview:\n", train.head())
print("\nTest preview:\n", test.head())
print("\nTrain columns:\n", train.columns)
print("\nTest columns:\n", test.columns)
print("\nTrain info:")
train.info()
print("\nMissing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())
print("\nNumeric summary:\n", train.describe())
print("\nOverall survival rate:\n", train["Survived"].value_counts(normalize=True))
print("\nSurvival rate by sex:\n", train.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by passenger class:\n", train.groupby("Pclass")["Survived"].mean())
print("\nSurvival rate by embarkation port:\n", train.groupby("Embarked")["Survived"].mean())

# These two plots are simple first-pass visuals for a beginner.
train.groupby("Sex")["Survived"].mean().plot(kind="bar", title="Survival Rate by Sex")
plt.tight_layout()
plt.show()

train["Age"].plot(kind="hist", bins=20, title="Age Distribution")
plt.tight_layout()
plt.show()

# Modeling ------------------------------------------------
# Start with a small feature set that is commonly useful for Titanic.
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train[features]
y = train["Survived"]
X_test = test[features]

# Numeric and categorical columns need different preprocessing.
numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
categorical_features = ["Sex", "Embarked"]

# Fill missing numeric values with the median.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Fill missing categorical values, then convert categories into numeric columns.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the right transformation to each column group.
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# The pipeline keeps preprocessing and model training together.
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Hold out 20% of the training data to estimate how well the model generalizes.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the baseline model and print standard evaluation metrics.
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
print("\nValidation accuracy:", accuracy_score(y_val, val_preds))
print("\nConfusion matrix:\n", confusion_matrix(y_val, val_preds))
print("\nClassification report:\n", classification_report(y_val, val_preds))

# Retrain using the full labeled dataset before predicting the unseen test data.
model.fit(X, y)
test_preds = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission file created: {SUBMISSION_PATH}")
