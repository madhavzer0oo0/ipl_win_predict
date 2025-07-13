# 🏏 IPL Win Probability Predictor

This project is a machine learning web app that predicts the probability of a team winning an IPL (Indian Premier League) match based on live match conditions. It’s built using **scikit-learn**, **Streamlit**, and real IPL ball-by-ball data from Kaggle.

### 🔍 What It Does
- Takes real-time match input: batting team, bowling team, target score, current score, wickets lost, and overs completed.
- Calculates derived features like `runs_left`, `balls_left`, `wickets_left`.
- Uses a trained **Logistic Regression** model to predict the chance of the batting team winning.
- Displays the prediction live with an interactive web interface.

---

## 🖥️ Tech Stack

| Layer         | Tools / Libraries                        |
|---------------|------------------------------------------|
| Data          | IPL 2008–2023 dataset from Kaggle        |
| ML Model      | scikit-learn Logistic Regression         |
| Web UI        | Streamlit                                |
| Deployment    | Streamlit Cloud                          |

---

## 🧠 How It Works

1. **Input**:
   - Batting/Bowling Team
   - Venue
   - Target Score
   - Current Score
   - Overs Completed
   - Wickets Lost

2. **Features Computed**:
   - `runs_left` = target - current_score
   - `balls_left` = 120 - (overs × 6)
   - `wickets_left` = 10 - wickets

3. **Model Prediction**:
   - Returns win probability for the batting team

---


