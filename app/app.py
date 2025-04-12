import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="NeuroScan.AI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import tempfile
from datetime import datetime
from model_utils import load_model, preprocess_image, get_class_names
from treatment_utils import get_treatment_recommendations, format_recommendations
from pdf_generator import ReportGenerator
from treatment_recommendations import get_recommendations
import google.generativeai as genai
import json
from usage_guide import get_welcome_text, get_usage_steps, get_disclaimer, get_help_text

# Configure Gemini
try:
    genai.configure(api_key="AIzaSyDeCmD3xtkED4fGUpvcfUY2-cSmemlyZY4")
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error(f"Configuration Error: {str(e)}")
    model = None

def get_gemini_recommendations(predicted_class, confidence, patient_data, probabilities):
    """Get detailed, personalized treatment recommendations from Gemini."""
    if not model:
        st.warning("Gemini model not available, using fallback recommendations")
        return get_recommendations(predicted_class, confidence)
    
    try:
        # Debug logging
        st.write("Debug: Creating MRI analysis...")
        
        # Create a detailed analysis of the MRI results
        mri_analysis = (
            f"MRI SCAN ANALYSIS:\n"
            f"Primary Finding: {predicted_class}\n"
            f"Confidence Level: {confidence:.2f}%\n\n"
            f"Detailed Probability Analysis:\n"
        )
        
        for tumor_type, prob in probabilities.items():
            prob_fixed = prob * 100 if prob <= 1 else prob
            mri_analysis += f"- {tumor_type}: {prob_fixed:.2f}%\n"
        
        # Format patient information
        patient_info = "PATIENT PROFILE:\n"
        for section, data in patient_data.items():
            if data:
                patient_info += f"\n{section}:\n"
                for key, value in data.items():
                    if value:
                        patient_info += f"- {key}: {value}\n"
        
        # Create the prompt text
        prompt_text = (
            "As a medical AI specialist, analyze this case and provide a comprehensive treatment plan.\n\n"
            f"{mri_analysis}\n"
            f"{patient_info}\n"
            "Provide a detailed treatment plan with these sections:\n\n"
            "1. IMMEDIATE RECOMMENDATIONS\n"
            "2. TREATMENT PLAN\n"
            "3. MONITORING & FOLLOW-UP\n"
            "4. LIFESTYLE MODIFICATIONS\n"
            "5. SUPPORT & RESOURCES\n\n"
            "Keep recommendations evidence-based and specific to the patient's condition."
        )
        
        # Debug logging
        st.write("Debug: Sending request to Gemini...")
        
        try:
            # Pass the prompt text directly to generate_content
            response = model.generate_content(prompt_text)
            st.write("Debug: Received response from Gemini")
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif isinstance(response, dict) and 'candidates' in response:
                return response['candidates'][0]['content']['parts'][0]['text']
            else:
                st.error("Unexpected response format from Gemini")
                return get_recommendations(predicted_class, confidence)
                
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            st.write("Debug: Falling back to basic recommendations")
            return get_recommendations(predicted_class, confidence)
            
    except Exception as e:
        st.error(f"Error in recommendation generation: {str(e)}")
        st.write("Debug: Exception traceback:", e.__traceback__)
        return get_recommendations(predicted_class, confidence)

def generate_report(image_path, patient_data, predicted_class, confidence, probabilities):
    report = ReportGenerator()
    report.add_header()
    report.add_patient_info(patient_data)
    report.add_mri_image(image_path)
    
    # Convert probabilities to percentages
    prob_percentages = {k: v * 100 for k, v in probabilities.items()}
    report.add_classification_results(predicted_class, confidence * 100, prob_percentages)
    
    try:
        # Get personalized recommendations from Gemini
        recommendations = get_gemini_recommendations(
            predicted_class, 
            confidence * 100, 
            patient_data,
            probabilities
        )
        
        # Debug logging
        st.write("Debug: Recommendations type:", type(recommendations))
        
        # Ensure recommendations is a string
        if isinstance(recommendations, list):
            recommendations = "\n".join(recommendations)
        
        report.add_treatment_recommendations(recommendations)
        
    except Exception as e:
        st.error(f"Error in report generation: {str(e)}")
        # Fallback to basic recommendations
        basic_recommendations = get_recommendations(predicted_class, confidence * 100)
        report.add_treatment_recommendations(basic_recommendations)
    
    report.add_disclaimer()
    
    # Save report with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"NueroScan_Report_{timestamp}.pdf"
    report.generate_report(output_path)
    return output_path

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .treatment-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("NeuroScan.AI 🧠")
st.markdown("""
    ### Brain Tumor Classification System
    Upload a brain MRI scan and provide patient information for a comprehensive analysis and treatment recommendations.
    """)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Add after the title, before the main content
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# Add sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Main Application"):
        st.session_state.page = 'main'
    if st.button("How to Use"):
        st.session_state.page = 'help'

# Main content logic
if st.session_state.page == 'help':
    st.markdown(get_welcome_text())
    
    # Display usage steps
    for step in get_usage_steps():
        st.subheader(step["title"])
        st.markdown(step["description"])
        st.markdown("---")
    
    # Display disclaimer
    st.markdown(get_disclaimer())
    
    # Display help text
    st.markdown(get_help_text())
    
else:  # Main application page
    # Step 1: Upload Image
    if st.session_state.step == 1:
        st.markdown("### Step 1: Upload MRI Scan")
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.session_state.uploaded_image = image
            
            if st.button("Proceed to Patient Information"):
                st.session_state.step = 2
                st.experimental_rerun()

    # Step 2: Patient Information
    elif st.session_state.step == 2:
        st.markdown("### Step 2: Patient Information")
        
        # Create a dictionary to store all patient information
        patient_data = {}
        
        with st.expander("Demographics", expanded=True):
            demographics = {
                "Age": st.number_input("Age", min_value=0, max_value=120, value=30),
                "Gender": st.selectbox("Gender", ["", "Male", "Female", "Other"]),
                "Weight (kg)": st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0),
                "Height (cm)": st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=170.0),
                "Ethnicity": st.selectbox("Ethnicity", ["", "Asian", "African", "Caucasian", "Hispanic", "Other"])
            }
            patient_data["Demographics"] = {k: v for k, v in demographics.items() if v}
        
        with st.expander("Medical History", expanded=True):
            medical_history = {
                "Family History": st.text_area("Family History of Cancer/Tumors"),
                "Current Medications": st.text_area("Current Medications"),
                "Allergies": st.text_area("Known Allergies"),
                "Previous Surgeries": st.text_area("Previous Surgeries"),
                "Existing Conditions": st.text_area("Existing Medical Conditions")
            }
            patient_data["Medical History"] = {k: v for k, v in medical_history.items() if v}
        
        with st.expander("Symptoms", expanded=True):
            symptoms = {
                "Duration": st.text_input("Duration of Symptoms"),
                "Primary Symptoms": st.text_area("Primary Symptoms"),
                "Symptom Severity (1-10)": st.slider("Symptom Severity", 1, 10, 5),
                "Recent Changes": st.text_area("Recent Changes in Symptoms")
            }
            patient_data["Symptoms"] = {k: v for k, v in symptoms.items() if v}
        
        with st.expander("Lifestyle Factors", expanded=True):
            lifestyle = {
                "Smoking Status": st.selectbox("Smoking Status", ["", "Never", "Former", "Current"]),
                "Alcohol Consumption": st.selectbox("Alcohol Consumption", ["", "None", "Occasional", "Regular"]),
                "Physical Activity": st.selectbox("Physical Activity Level", ["", "Sedentary", "Light", "Moderate", "Active"]),
                "Occupation": st.text_input("Occupation"),
                "Stress Level (1-10)": st.slider("Stress Level", 1, 10, 5)
            }
            patient_data["Lifestyle"] = {k: v for k, v in lifestyle.items() if v}
        
        with st.expander("Treatment History", expanded=True):
            treatment_history = {
                "Previous Cancer Treatments": st.text_area("Previous Cancer Treatments"),
                "Current Treatments": st.text_area("Current Treatments"),
                "Alternative Therapies": st.text_area("Alternative Therapies Tried")
            }
            patient_data["Treatment History"] = {k: v for k, v in treatment_history.items() if v}
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Analysis"):
                st.session_state.patient_data = patient_data
                st.session_state.step = 3
                st.experimental_rerun()

    # Step 3: Results and Recommendations
    elif st.session_state.step == 3:
        st.markdown("### Step 3: Analysis Results")
        
        # Process image and get predictions
        if st.session_state.predictions is None:
            processed_image = preprocess_image(st.session_state.uploaded_image)
            st.session_state.predictions = st.session_state.model.predict(processed_image)
        
        # Get class names and predictions
        class_names = get_class_names()
        predictions = st.session_state.predictions
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
        # Create probability dictionary for PDF
        probabilities = {class_name: float(prob * 100) for class_name, prob in zip(class_names, predictions[0])}
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Classification Results")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            # Display probabilities
            for class_name, prob in probabilities.items():
                st.progress(prob/100, text=f"{class_name}: {prob:.2f}%")
            
            st.markdown(f"""
                #### 🎯 Final Prediction
                **{predicted_class}** with {confidence:.2f}% confidence
                """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💊 Treatment Recommendations")
            st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
            
            # Generate and display treatment recommendations using Gemini
            with st.spinner('Generating personalized treatment recommendations...'):
                gemini_recommendations = get_gemini_recommendations(
                    predicted_class,
                    confidence,
                    st.session_state.patient_data,
                    probabilities
                )
                st.markdown(gemini_recommendations)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and offer PDF download
        st.markdown("### 📄 Download Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                # Create temporary file for the uploaded image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    st.session_state.uploaded_image.save(tmp_file.name)
                    
                    # Generate PDF
                    output_path = generate_report(tmp_file.name, st.session_state.patient_data, predicted_class, confidence, probabilities)
                    
                    # Offer download
                    with open(output_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_file,
                            file_name=os.path.basename(output_path),
                            mime="application/pdf"
                        )
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
        
        # Option to start over
        if st.button("Start New Analysis"):
            st.session_state.step = 1
            st.session_state.uploaded_image = None
            st.session_state.predictions = None
            st.session_state.patient_data = None
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Developed with ❤️ using TensorFlow and Streamlit</p>
            <p>For educational and research purposes only</p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses a MobileNetV2-based deep learning model trained on brain MRI images.
        
        **Model Accuracy:** ~92%
        
        **Note:** This tool is for educational purposes only and should not be used for medical diagnosis.
        Always consult healthcare professionals for medical advice.
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. Upload a brain MRI image (JPG, JPEG, or PNG)
        2. Wait for the model to process the image
        3. View the prediction results and confidence scores
        """)

        st.header("Citations")
        st.markdown("""
        This application was developed using the following resources:

        1. **Dataset Source:**
           - Sartaj Bhuvaji et al. (2020)
           - Brain Tumor Classification (MRI)
           - [Kaggle Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

        2. **Model Development Reference:**
           - Amirhosein Mousavian (2020)
           - Brain Tumor Detection
           - [Kaggle Notebook](https://www.kaggle.com/code/amirhoseinmousavian/brain-tumor-detection-70-accuracy/notebook)

        The current implementation has been significantly enhanced with additional features and improvements.
        """) 