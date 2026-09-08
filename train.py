"""Train a baseline churn prediction model and save it as a reusable pipeline."""
from pathlib import Path

import joblib   # to save the model
import pandas as pd     # for data manipulation
from sklearn.compose import ColumnTransformer   # to apply different preprocessing to different columns
from sklearn.linear_model import LogisticRegression     # our baseline model
from sklearn.metrics import classification_report, roc_auc_score    # to evaluate the model
from sklearn.model_selection import train_test_split    # to split the data into train and test sets
from sklearn.pipeline import Pipeline   # to create a reusable pipeline into one object
from sklearn.preprocessing import OneHotEncoder      # to convert categorical variables into numerical variables


# load and clean the data
df = pd.read_csv("data/Telco_Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)   # convert TotalCharges to numeric, setting errors to NaN and fill them with 0
df = df.drop(columns=["customerID"])

# seperate features and target and split the data into train and test sets
X = df.drop(columns=["Churn"]) # target variable is Churn
y = (df["Churn"] == "Yes").astype(int)  # convert target variable to binary (1 for Yes, 0 for No)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# identify categorical and numerical columns
numerical_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [
    col for col in X.columns if col not in numerical_cols
    ]

# build the ColumnTransformer to preprocess X, names preprocessor

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_cols),
     ], remainder="passthrough"
     )

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

Path("models").mkdir(exist_ok=True)  # create models directory if it doesn't exist
joblib.dump(pipeline, "models/churn_model.joblib")  # save the model as a reusable pipeline