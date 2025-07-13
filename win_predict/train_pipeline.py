# win_pred/train_pipeline.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from skops.io import dump

# Load dataset
df = pd.read_csv("win_prediction_data.csv")

# Features & target
X = df.drop(columns=["won"])
y = df["won"]

# Define categorical columns
categorical_cols = ["batting_team", "bowling_team", "venue"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

# Train the model
pipeline.fit(X, y)

# Save as skops
dump(pipeline, "final_win_predictor_pipeline.skops")
