import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set up the Streamlit page
st.set_page_config(
    page_title="Text Classification App",
    page_icon="🤖",
    layout="centered"
)

# Add a title and description
st.title("Text Classification App 🎯")
st.write("Enter your text below to classify it!")

# Create the text input
user_input = st.text_area("Enter text to classify:", height=150)

# Add a button to trigger classification
if st.button("Classify Text"):
    if user_input:
        # Vectorize the input text
        input_vector = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(input_vector)[0]
        
        # Show the prediction with some styling
        st.success(f"Prediction: {prediction}")
        
        # Add confidence score if available (for models that support predict_proba)
        try:
            confidence = model.predict_proba(input_vector).max()
            st.info(f"Confidence: {confidence:.2%}")
        except:
            pass
    else:
        st.warning("Please enter some text to classify!")

# Add some helpful information in the sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses a machine learning model to classify text.
    
    How to use:
    1. Enter your text in the text area
    2. Click the 'Classify Text' button
    3. View the prediction and confidence score
    """
)
