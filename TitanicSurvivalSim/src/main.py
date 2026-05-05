"""
Titanic survival — feature engineering + model comparison

Steps:
1. Load train/test CSVs.
2. Explore the data: missing values, survival rates, correlation heatmap.
3. Engineer new features from existing columns (Title, FamilySize, etc.)
4. Preprocess with a scikit-learn pipeline.
5. Compare Logistic Regression vs Random Forest on a validation split.
6. Use the better model to write the submission CSV.

Key idea: correlation analysis tells us HOW MUCH each column is related
to survival (values range from -1 to +1). Positive = tends to survive,
negative = tends to die. Features near 0 are basically noise.

Feature engineering means creating NEW columns from existing ones that
capture patterns a model can learn more easily. For example, the raw
Name column is useless, but the Title extracted from it (Mr, Mrs, Miss)
is very predictive.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Build paths relative to this file so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nMissing values in train:\n", train.isnull().sum())
print("\nSurvival rate by sex:\n", train.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by class:\n", train.groupby("Pclass")["Survived"].mean())

# ---------------------------------------------------------------------------
# 2. CORRELATION ANALYSIS
# Correlation only works on numbers, so we encode Sex as 0/1 first.
# The result shows how much each column moves TOGETHER with Survived.
# ---------------------------------------------------------------------------
corr_df = train.copy()
corr_df["Sex_num"] = (corr_df["Sex"] == "female").astype(int)  # female=1, male=0

numeric_cols = ["Survived", "Pclass", "Sex_num", "Age", "SibSp", "Parch", "Fare"]
corr_matrix = corr_df[numeric_cols].corr()

print("\nCorrelation with Survived:\n",
      corr_matrix["Survived"].sort_values(ascending=False))

# Visual heatmap — darker = stronger relationship (positive or negative).
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix\n(values close to ±1 = strong relationship with Survived)")
plt.tight_layout()
plt.show()

# Bar chart of absolute correlation with Survived (magnitude = importance proxy).
corr_with_target = corr_matrix["Survived"].drop("Survived").abs().sort_values(ascending=True)
corr_with_target.plot(kind="barh", title="Feature correlation with Survived (absolute)")
plt.xlabel("Pearson |r|")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# We create new columns on BOTH train and test so they match at prediction time.
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that capture patterns raw columns miss."""
    df = df.copy()

    # Title: extracted from the Name field using a regex.
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = {"Rev", "Dr", "Col", "Major", "Mlle", "Countess",
                   "Ms", "Lady", "Jonkheer", "Don", "Dona", "Capt", "Sir"}
    df["Title"] = df["Title"].replace(list(rare_titles), "Rare")

    # FamilySize and IsAlone.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Has_Cabin.
    df["Has_Cabin"] = df["Cabin"].notna().astype(int)

    return df


train = engineer_features(train)
test = engineer_features(test)

print("\nNew feature preview (train):\n",
      train[["Name", "Title", "FamilySize", "IsAlone", "Has_Cabin"]].head(10))
print("\nTitle value counts:\n", train["Title"].value_counts())
print("\nSurvival rate by Title:\n", train.groupby("Title")["Survived"].mean())
print("\nSurvival rate by FamilySize:\n", train.groupby("FamilySize")["Survived"].mean())

# ---------------------------------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

# We now include the engineered features.
numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch", "FamilySize",
                    "IsAlone", "Has_Cabin"]
categorical_features = ["Sex", "Embarked", "Title"]

features = numeric_features + categorical_features
X = train[features]
y = train["Survived"]
X_test = test[features]

# Median imputation handles the few missing Age/Fare values.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Mode imputation + one-hot encoding for text columns.
# OneHotEncoder turns each category into its own 0/1 column.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------------------------------------------------------
# 5. TUNE RANDOM FOREST with GridSearchCV
# Instead of guessing hyperparameters, we try many combinations and pick
# the best one automatically.
#
# n_estimators: number of trees — more = more stable
# max_depth: how deep each tree grows — shallower = less overfitting
# min_samples_leaf: minimum samples required at a leaf — higher = smoother
# max_features: how many features each tree can see — "sqrt" is standard RF
#
# GridSearchCV tests every combination using 5-fold cross-validation so
# the winner is evaluated fairly, not just on one lucky split.
# ---------------------------------------------------------------------------
base_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [4, 5, 6],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt"],
}

print("\nRunning GridSearchCV (this takes ~30 seconds)...")
grid_search = GridSearchCV(base_rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X, y)

print("Best params:", grid_search.best_params_)
print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# Quick sanity check on a held-out split.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_rf.fit(X_train, y_train)
val_preds = best_rf.predict(X_val)
print("\nValidation accuracy (held-out split):", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
rf_model = best_rf.named_steps["classifier"]
ohe_cols = (best_rf.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features))
all_feature_names = numeric_features + list(ohe_cols)

importance_series = pd.Series(rf_model.feature_importances_, index=all_feature_names)
importance_series = importance_series.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance_series.tail(15).plot(kind="barh")
plt.title("Random Forest (tuned) — Top 15 Feature Importances")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 7. GENERATE SUBMISSION — retrain best RF on ALL labeled data
# ---------------------------------------------------------------------------
best_rf.fit(X, y)
test_preds = best_rf.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission file created: {SUBMISSION_PATH}")
