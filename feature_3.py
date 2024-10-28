import os
import openai
import streamlit as st
from openai import OpenAI

st.title("Solutions for your water")
st.header("Potential Solution to improve your water quality")

# Set up API key
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

# Wrapper function for OpenAI API
def get_completion(prompt, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "Your job is to suggest me 5 solutions that I can do immediately to make my water drinkable"},
            {"role": "user",
             "content": prompt},
        ]
    )
    return completion.choices[0].message.content

# Ideal ranges for each water quality indicator
IDEAL_RANGES = {
    "pH": (6.5, 8.5),
    "Chlorine (mg/L)": (0, 4),
    "Hardness (mg/L as CaCO3)": (0, 120),
    "Nitrates (mg/L)": (0, 10),
    "Lead (µg/L)": (0, 15)
}

# Evaluate each indicator against ideal range
def evaluate_indicator(value, ideal_range):
    if ideal_range[0] <= value <= ideal_range[1]:
        return "within the ideal range."
    elif value < ideal_range[0]:
        return "below the ideal range."
    else:
        return "above the ideal range."

# Water quality check function to get and evaluate user input
def water_quality_check():
    readings = {
        "pH": st.number_input("Enter pH level:", min_value=0.0, max_value=14.0, step=0.1),
        "Chlorine (mg/L)": st.number_input("Enter Chlorine (mg/L):", min_value=0.0, max_value=10.0, step=0.1),
        "Hardness (mg/L as CaCO3)": st.number_input("Enter Hardness (mg/L as CaCO3):", min_value=0.0, max_value=500.0, step=1.0),
        "Nitrates (mg/L)": st.number_input("Enter Nitrates (mg/L):", min_value=0.0, max_value=50.0, step=0.1),
        "Lead (µg/L)": st.number_input("Enter Lead (µg/L):", min_value=0.0, max_value=100.0, step=0.1)
    }

    # Prepare the formatted results
    results_prompt = "Water Quality Results:\n"
    for indicator, value in readings.items():
        range_feedback = evaluate_indicator(value, IDEAL_RANGES[indicator])
        results_prompt += f"- {indicator}: {value} is {range_feedback}\n"
    
    return results_prompt

# Create Streamlit form
with st.form(key="chat"):
    # Run the water quality check and prepare the prompt
    results_prompt = water_quality_check()
    
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Display water quality results
        st.text_area("Water Quality Results", value=results_prompt, height=200)
        
        # Fetch and display the AI's response
        ai_response = get_completion(results_prompt)
        st.write("AI Suggestions:")
        st.write(ai_response)
