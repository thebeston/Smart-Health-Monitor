import streamlit as st
import pandas as pd
from training import *
from generative_bot import generate_response, health_chatbot, clean_response

def display():
    # Page Title and Intro
    st.title("🏥 Smart Health Monitor")
    st.write("### Track your health metrics and receive personalized tips.")

    # Section: General Information
    st.header("👤 General Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=20)
    with col2:
        gender = st.selectbox("Gender", ("Male", "Female"))

    # Section: Height and Weight
    st.header("📏 Height & Weight")
    st.subheader("Enter your height:")
    col3, col4 = st.columns(2)
    with col3:
        feet = st.number_input("Feet", min_value=0, max_value=10, step=1, value=5)
    with col4:
        inches = st.number_input("Inches", min_value=0, max_value=11, step=1, value=7)

    weight = st.number_input("Weight (Lbs)", min_value=30, max_value=500, step=1, value=130)

    # Section: Physical Activity
    st.header("🏋️ Physical Activity")
    col5, col6 = st.columns(2)
    with col5:
        steps = st.number_input("Daily Steps", min_value=0, step=100, value=5000)
    with col6:
        intensity = st.slider("Workout Intensity (1 - lowest, 4 - highest)", 0, 4, 2)

    workout_duration = st.number_input("Workout Duration (mins per day)", min_value=0, max_value=1200, step=5, value=20)

    # Section: Diet
    st.header("🥗 Diet")
    diet = st.text_area(
        "Describe your daily diet (Include dietary resctrictions if any)",
        placeholder="E.g., Breakfast: Oatmeal, Lunch: Salad, Dinner: Grilled Chicken..."
    )

    # Section: Sleep
    st.header("🛌 Sleep")
    sleep_duration = st.number_input(
        "How many hours do you sleep on average?", min_value=0.0, max_value=24.0, step=0.5, value=7.0
    )
    sleep_quality = st.slider("Sleep Quality (1 - worst, 10 - best)", 0, 10, 6)

    # Section: Mental Health
    st.header("🧠 Mental Health")
    stress_level = st.slider("Stress Level (1 - least stressed, 10 - most stressed)", 0, 10, 5)
    mental_evaluation = st.text_area(
        "Describe your current mental state",
        placeholder="E.g., Feeling stressed due to work deadlines..."
    )

    # Generate Button
    if st.button("Generate Health Tips 💡"):
        # Calculations for predictions
        height_meters = (feet * 12 + inches) * 0.0254
        weight_kg = float(weight) * 0.453592

        obesity_data = {
            'Age': age,
            'Gender': gender,
            'Height': height_meters,
            'Weight': weight,
            'BMI': weight_kg / (height_meters ** 2),
            'PhysicalActivityLevel': intensity,
        }

        sleep_data = {
            'Age': age,
            'Gender': gender,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': sleep_quality,
            'Physical Activity Level': workout_duration,
            'Stress Level': stress_level,
            'Daily Steps': steps,
        }

        mental_health_description = {'statement': mental_evaluation}

        # DataFrames for Prediction Models
        obesity_predictor = pd.DataFrame([obesity_data])
        sleep_predictor = pd.DataFrame([sleep_data])
        mental_health_predictor = pd.DataFrame([mental_health_description])

        # Load Models and Make Predictions
        obesity_model, obesity_features = train_and_predict_obesity_status()
        sleep_model, sleep_features = train_and_predict_sleep_disorder()
        mental_health_model, vectorizer = train_and_predict_mental_health_state()

        obesity_prediction = get_prediction(obesity_model, obesity_predictor, obesity_features)
        sleep_prediction = get_prediction(sleep_model, sleep_predictor, sleep_features,
                                          merge_columns=['BMI Category'], merge_values=obesity_prediction)
        mental_health_prediction = get_prediction(mental_health_model, mental_health_predictor, vec=vectorizer)

        # Generate Response using the Chatbot
        response = health_chatbot(obesity_data, sleep_data, obesity_prediction, diet, sleep_prediction, mental_health_prediction)
        cleaned_response = clean_response(response)

        
        
        # Display the response in a non-editable text area
        st.text_area("💡 Health Tips", value=cleaned_response, height=300, disabled=True)

# Run the display function to start the app
display()
