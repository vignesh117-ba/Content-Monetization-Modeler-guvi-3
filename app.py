import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("lasso_model.pkl")

st.title("📊 YouTube Ad Revenue Predictor")

# -------- USER INPUTS --------

views = st.number_input("Views", min_value=0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
watch_time = st.number_input("Watch Time (minutes)", min_value=0.0)
video_length = st.number_input("Video Length (minutes)", min_value=0.0)
subscribers = st.number_input("Subscribers", min_value=0)

engagement_rate = st.number_input("Engagement Rate", min_value=0.0)
watch_ratio = st.number_input("Watch Ratio", min_value=0.0)

category = st.selectbox(
    "Select Category",
    ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
)

# -------- PREDICTION --------

if st.button("Predict Revenue"):

    input_dict = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time,
        "video_length_minutes": video_length,
        "subscribers": subscribers,
        "engagement_rate": engagement_rate,
        "watch_ratio": watch_ratio,

        "category_Entertainment": 0,
        "category_Gaming": 0,
        "category_Lifestyle": 0,
        "category_Music": 0,
        "category_Tech": 0
    }

    input_dict[f"category_{category}"] = 1

    input_data = pd.DataFrame([input_dict])

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Ad Revenue: ${prediction[0]:.2f}")
    
    
st.write("Expected Features:")
st.write(model.feature_names_in_)