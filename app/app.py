import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from model_utils import load_model, preprocess_image, get_class_names

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# Load model (cache it to avoid reloading)
@st.cache_resource
def get_model():
    return load_model()

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = get_model()

# Title and description
st.title("🧠 Brain Tumor Classification")
st.markdown("""
This application uses a deep learning model to classify brain MRI images into four categories:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

Upload a brain MRI image to get a prediction.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = st.session_state.model.predict(processed_image)
    
    # Get class names
    class_names = get_class_names()
    
    # Display results
    st.subheader("Prediction Results")
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        # Show confidence scores
        st.write("Confidence Scores:")
        for i, (class_name, confidence) in enumerate(zip(class_names, predictions[0])):
            st.progress(float(confidence), text=f"{class_name}: {confidence:.2%}")
    
    with col2:
        # Show final prediction
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2%}")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses a MobileNetV2-based deep learning model trained on brain MRI images.
    
    **Model Accuracy:** ~72%
    
    **Note:** This tool is for educational purposes only and should not be used for medical diagnosis.
    Always consult healthcare professionals for medical advice.
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Upload a brain MRI image (JPG, JPEG, or PNG)
    2. Wait for the model to process the image
    3. View the prediction results and confidence scores
    """) 