import os
import openai
import streamlit as st
import easyocr
import numpy as np
from PIL import Image

# Set up API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify language for OCR

# Wrapper function to get suggestions from OpenAI
def get_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Your job is to suggest 5 immediate solutions to improve water quality."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content'].strip()

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    img = np.array(image)
    results = reader.readtext(img)
    extracted_text = " ".join([text for _, text, _ in results])
    return extracted_text

# Define ideal ranges for each water quality indicator
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

# Function to prepare prompt based on manual input
def prepare_manual_input_prompt(readings):
    prompt = "Water Quality Results:\n"
    for indicator, value in readings.items():
        range_feedback = evaluate_indicator(value, IDEAL_RANGES[indicator])
        prompt += f"- {indicator}: {value} is {range_feedback}\n"
    return prompt

# Set up Streamlit interface
st.title("Water Quality Improvement Suggestions")
st.header("Choose Input Method")

# User selects input method
input_option = st.radio("Select Input Method:", ("Manual Input", "Image Upload"))

if input_option == "Manual Input":
    # Collect manual inputs
    readings = {
        "pH": st.number_input("Enter pH level:", min_value=0.0, max_value=14.0, step=0.1),
        "Chlorine (mg/L)": st.number_input("Enter Chlorine (mg/L):", min_value=0.0, max_value=10.0, step=0.1),
        "Hardness (mg/L as CaCO3)": st.number_input("Enter Hardness (mg/L as CaCO3):", min_value=0.0, max_value=500.0, step=1.0),
        "Nitrates (mg/L)": st.number_input("Enter Nitrates (mg/L):", min_value=0.0, max_value=50.0, step=0.1),
        "Lead (µg/L)": st.number_input("Enter Lead (µg/L):", min_value=0.0, max_value=100.0, step=0.1)
    }
    
    # Prepare prompt based on manual inputs
    results_prompt = prepare_manual_input_prompt(readings)
    
    if st.button("Analyze"):
        # Display water quality results
        st.text_area("Water Quality Results", value=results_prompt, height=200)
        
        # Fetch and display AI suggestions
        ai_response = get_completion(results_prompt)
        st.subheader("AI Suggestions:")
        st.write(ai_response)

elif input_option == "Image Upload":
    # Upload image
    uploaded_file = st.file_uploader("Upload an image of your water test kit results...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract text from image
        with st.spinner("Extracting text from the image..."):
            extracted_text = extract_text_from_image(image)
        
        # Display extracted text
        st.subheader("Extracted Text")
        st.write(extracted_text)
        
        # Analyze extracted text with OpenAI
        if extracted_text:
            with st.spinner("Analyzing extracted text with OpenAI..."):
                analysis_prompt = f"Water Quality Results from image:\n\n{extracted_text}\n\nProvide 5 solutions to improve water quality."
                analysis = get_completion(analysis_prompt)
            
            # Display OpenAI analysis
            st.subheader("AI Suggestions:")
            st.write(analysis)
        else:
            st.write("No text found in the image. Please try another image.")
