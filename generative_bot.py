import os
import openai
from dotenv import load_dotenv
from training import *

load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

def generate_response(prompt):
    try:
        completion = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a professional health mentor that is giving health tips to college students and adults"},
            {"role": "system", "content": "You give tips based on the metrics that are entered about the user's health"},
            {"role": "system", "content": "Make sure to keep the tips specific to the metrics entered into the prompt"},
            {"role": "system", "content": "You give specific tips that are useful."},
            {"role": "user", "content": prompt},
        ])
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error has occured while generating the response: {e}")
        return None
    
def clean_response(response):
    # Split the response into sections using double newlines as separators
    sections = response.strip().split('\n\n')
    
    # Format the sections using Markdown, adding bullet points or bold text
    formatted_response = ""
    for section in sections:
        # Detect if the section is a tip list (starts with a number or bullet point)
        if section[0].isdigit() or section.startswith('-'):
            formatted_response += f"- {section}\n"
        else:
            # Treat the first line as a header and the rest as body text
            lines = section.split('\n', 1)
            header = f"{lines[0]}"  # Convert first line to a Markdown header
            body = lines[1] if len(lines) > 1 else ""
            formatted_response += f"{header}\n\n{body}\n\n"
    
    return formatted_response

def health_chatbot(obesity_data, sleep_data, obesity_prediction, sleep_prediction, diet, mental_health_prediction):
    Age = obesity_data['Age']
    Gender = obesity_data['Gender']
    obesity_status = obesity_prediction[0]
    sleep_time = sleep_data['Sleep Duration']
    sleep_quality = sleep_data['Quality of Sleep']
    sleep_disorder = sleep_prediction[0] if sleep_prediction and sleep_prediction[0] != 'None' else 'None'
    stress_level = sleep_data['Stress Level']
    exercize_time = sleep_data['Physical Activity Level']
    daily_steps = sleep_data['Daily Steps']
    mental_condition = mental_health_prediction[0]

    prompt = (
        f"I am {Age} years old. "
        f"I am a {Gender}. "
        f"I am {obesity_status} and I want to be Normal Weight. "
        f"I sleep for {sleep_time} hours. "
        f"This is a possible sleep disorder that I might have: {sleep_disorder}. If this is none, completly neglect this section in the response. "
        f"If the sleep disorder is anything other than none, then give tips geared towards improving this sleep disorder. "
        f"On a scale from 1 to 10, my level of stress is {stress_level}. Give tips on how to reduce the stress level. "
        f"I exercise for {exercize_time} per day on average. Determine if this is a good amount of workout and recommend some typical workout tips based on the level. "
        f"This is the description of my diet on an average day: {diet}. Can you determine if this diet is good enough and tell me how to improve it if it's not? Also account for dietary restrictions if they are mentioned. If there is nothing mentioned here or the description isn't related to a diet at all, don't generate thi section. "
        f"I walk around {daily_steps} steps per day. "
        f"My mental state right now is {mental_condition}. Provide assistance on how to reduce the current mental state here to normal if it isn't normal. If it is normal, then just say that my mental state is fine and give general tips to keep it up. "
        f"Can you give me health tips that I could apply to improve these features that I described above? For each response, give tips based on how close the metrics are from the norm."
    )

    return generate_response(prompt)

if __name__ == "__main__":
    from input import user_input
    obesity_df, sleep_df, mental_health_df, diet = user_input()
    obesity_model, obesity_features = train_and_predict_obesity_status()
    sleep_model, sleep_features = train_and_predict_sleep_disorder()
    mental_health_model, vectorizer = train_and_predict_mental_health_state()

    obesity_prediction = get_prediction(obesity_model, obesity_df, obesity_features)
    sleep_prediction = get_prediction(sleep_model, sleep_df, sleep_features, merge_columns=['BMI Category'], merge_values=obesity_prediction)
    mental_health_prediction = get_prediction(mental_health_model, mental_health_df, vec=vectorizer)

    generated_response = health_chatbot(obesity_df, sleep_df, obesity_prediction, diet, sleep_prediction, mental_health_prediction)
    formated_response = clean_response(generated_response)
    print(formated_response)
    

    