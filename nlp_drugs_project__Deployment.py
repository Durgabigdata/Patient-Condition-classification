import streamlit as st
import pickle
import zipfile
import os


model_zip_path = "rf_model.zip"
model_path = "rf_model.pkl"
vectorizer_path = "vectorizer.pkl"

# Extract model if it's not already extracted
if not os.path.exists(model_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall()  # Extracts rf_model.pkl

# Check if extracted files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found after extraction!")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file {vectorizer_path} not found!")

# Load model and vectorizer
with open(vectorizer_path, 'rb') as vec_f:
    vectorizer = pickle.load(vec_f)

with open(model_path, 'rb') as model_f:
    model = pickle.load(model_f)


# Streamlit setup
st.title("Drug Classification NLP Model")
st.write("Predicts the Drug Category based on Patient reviews")


review = st.text_area("Enter the patient review:",placeholder="Enter your Review here...")


if st.button("Predict"):
    if review.strip():
       
        transformed_review = vectorizer.transform([review])
        
        
        prediction = rf_model.predict(transformed_review)
        
       
        st.write(f"The predicted drug category is: **{prediction[0]}**")
        
       
        
    else:
        st.warning("Please enter a valid review.")  


st.sidebar.write("### Model Information")
st.sidebar.write("This model was trained using Random Forest and is designed to classify patient reviews into categories: Depression, Diabetes, and High Blood Pressure.")


st.sidebar.write("OUTPUT")

st.sidebar.write("[100] means Depression")
st.sidebar.write("[010] means Diabetes")
st.sidebar.write("[001] means High Blood Pressure")

st.sidebar.write("Model Accuracy: 86%")

# Separator for better UI layout

st.markdown("---")

# Instructions for users

st.markdown('<div class="instructions"><h4>How It Works:</h4><ol><li>Enter a patient\'s review describing their condition.</li><li>Click the \'Predict\' button to predict the condition.</li><li>Get recommended drugs based on the classified condition.</li></ol></div>', unsafe_allow_html=True)

