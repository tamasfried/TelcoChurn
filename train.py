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

print(df.shape)