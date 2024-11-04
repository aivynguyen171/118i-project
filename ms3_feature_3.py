import os
import openai
import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image

# Set up Streamlit interface
st.title("Solutions for Your Water")
st.header("Potential Solutions to Improve Water Quality")

# Set up API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in your environment

# Wrapper function for OpenAI API using gpt-3.5-turbo
def get_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Your job is to suggest 5 solutions that I can do immediately to make my water drinkable."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content'].strip()

# Function to extract text from an image
def extract_text_from_image(image):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the language(s)
    
    # Convert uploaded image to OpenCV format
    img = np.array(image)
    
    # Perform OCR on the image
    results = reader.readtext(img)
    
    # Combine extracted text
    extracted_text = " ".join([text for _, text, _ in results])
    
    return extracted_text

# Function to analyze extracted text with OpenAI
def analyze_text_with_openai(text):
    # Define a prompt for OpenAI to analyze the extracted text
    prompt = f"Analyze the following text:\n\n{text}\n\nProvide a summary or insights on the content."
    
    # Send request to OpenAI's ChatCompletion API using gpt-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes text."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract text from the image
    with st.spinner("Extracting text from the image..."):
        extracted_text = extract_text_from_image(image)
    
    # Display extracted text
    st.subheader("Extracted Text")
    st.write(extracted_text)
    
    # Analyze the text with OpenAI if any text was extracted
    if extracted_text:
        with st.spinner("Analyzing text with OpenAI..."):
            analysis = analyze_text_with_openai(extracted_text)
        
        # Display OpenAI analysis
        st.subheader("OpenAI Analysis")
        st.write(analysis)
    else:
        st.write("No text found in the image.")
