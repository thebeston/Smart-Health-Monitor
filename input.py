import pandas as pd

def user_input():
    """Collect user input and split it into obesity and sleep metrics."""
    # Collecting user input data
    Age = int(input('Enter your Age: '))
    Gender = input('What is your Gender (Male/Female): ').strip().lower()
    Gender = 1 if Gender == 'male' else 2  # Convert to numeric

    Occupation = input("What is your occupation? ")

    # Get height in feet and inches
    feet = int(input('Feet: '))
    inches = float(input('Inches: '))
    Height = ((feet * 12 + inches) * 0.0254)  # Convert total inches to meters

    Weight = float(input("Weight (in lbs): ")) * 0.453592  # Convert to kg

    Physical_activity = int(input("Rate your physical activity level (1 - 4): "))
    Physical_activity_duration = int(input("How many minutes do you exercise daily on average? "))

    Daily_Steps = int(input("How many steps per day do you walk on average? "))

    Sleep_Duration = float(input("How many hours of sleep do you get on average? "))
    Sleep_Quality = int(input("Rate your quality of sleep (1 to 10): "))

    Stress_Level = int(input("How stressed are you (1 to 10)? "))
    Mental_Description = input("Describe your mental state: ")

    # Split input data into two dictionaries
    obesity_metrics = {
        'Age': Age,
        'Gender': Gender,
        'Height': Height,
        'Weight': Weight,
        'BMI': Weight/(Height**2),
        'PhysicalActivityLevel': Physical_activity,
    }

    sleep_metrics = {
        'Age': Age,
        'Gender': Gender,
        'Occupation': Occupation,
        'Sleep Duration': Sleep_Duration,
        'Quality of Sleep': Sleep_Quality,
        'Physical Activity Level': Physical_activity_duration,
        'Stress Level': Stress_Level,
        'Daily Steps': Daily_Steps,
    }

    mental_health_description = {
        'statement': Mental_Description
    }

    # Convert to DataFrames
    obesity_df = pd.DataFrame([obesity_metrics])
    sleep_df = pd.DataFrame([sleep_metrics])
    mental_health_df = pd.DataFrame([mental_health_description])

    return obesity_df, sleep_df, mental_health_df

# ========== Add Obesity Prediction to Sleep Data ==========
def add_obesity_to_sleep(sleep_df, obesity_prediction):
    """Add obesity prediction as a new feature in the sleep DataFrame."""
    sleep_df['BMI Category'] = obesity_prediction
    return sleep_df