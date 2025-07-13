# train_pipeline.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load your CSV
df = pd.read_csv("win_prediction_data.csv")

# Define input & output
X = df.drop(columns=["won"])
y = df["won"]

# Categorical columns
categorical_cols = ["batting_team", "bowling_team", "venue"]

# Preprocessor
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Train
pipeline.fit(X, y)

# Save to disk (in same folder)
joblib.dump(pipeline, "final_win_predictor_pipeline.pkl")
