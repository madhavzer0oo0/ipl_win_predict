import streamlit as st
import pandas as pd
from skops.io import get_untrusted_types
from skops.io import load



untrusted = get_untrusted_types(file="win_predict/final_win_predictor_pipeline.skops")
pipeline = load("final_win_predictor_pipeline.skops", trusted=untrusted)



# Predefined options
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
]

venues = [
    'Wankhede Stadium', 'Eden Gardens', 'Narendra Modi Stadium',
    'Arun Jaitley Stadium', 'MA Chidambaram Stadium', 'M Chinnaswamy Stadium'
]

# Page config
st.set_page_config(page_title="Cricket Win Predictor", page_icon="🏏")
st.title("🏏 IPL Win Probability Predictor")

st.markdown("#### Enter match situation below:")

# ⬇️ INPUTS (outside if block)
batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
venue = st.selectbox("Venue", venues)

target = st.number_input("Target Score", min_value=1)
current_score = st.number_input("Current Score", min_value=0, max_value=target)
wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10)
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

# ⬇️ Predict when button is clicked
if st.button("Predict Win Probability"):
    runs_left = target - current_score
    balls_left = 120 - int(overs_completed * 6)
    wickets_left = 10 - int(wickets_lost)

    if balls_left <= 0 or wickets_left <= 0:
        st.error("⚠️ Invalid input: Match cannot continue.")
    else:
        input_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'target': target
        }])

        # Predict win probability
        win_prob = pipeline.predict_proba(input_df)[0][1] * 100

        # Display
        st.metric(label=f"📊 Win Probability for {batting_team}", value=f"{win_prob:.2f}%")
