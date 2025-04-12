# NeuroScan.AI 🧠

NeuroScan.AI is an advanced brain tumor classification system that combines state-of-the-art deep learning with personalized medical recommendations. The system analyzes MRI scans, provides detailed classifications, and generates comprehensive medical reports with treatment recommendations.

## Features

- 🔍 **Advanced MRI Analysis**: Utilizes deep learning (MobileNetV2) to analyze brain MRI scans
- 📊 **Multi-Class Classification**: Identifies different types of brain tumors with confidence scores
- 👤 **Patient Information Integration**: Collects and incorporates detailed patient data for personalized analysis
- 🤖 **AI-Powered Recommendations**: Generates personalized treatment recommendations using Google's Gemini AI
- 📄 **Professional PDF Reports**: Creates comprehensive medical reports with:
  - Patient information
  - MRI scan analysis
  - Classification results with confidence scores
  - Personalized treatment recommendations
  - Space for doctor's approval and notes
- ✅ **Medical Validation**: Includes a section for doctor's review and approval

## Tech Stack

- TensorFlow 2.15.0 (Deep Learning Framework)
- Streamlit 1.32.0 (Web Application Framework)
- OpenCV 4.9.0.80 (Image Processing)
- Pillow 10.2.0 (Image Handling)
- NumPy 1.24.3 (Numerical Operations)

## Project Structure

```
NeuroScan.AI/
├── app/                    # Streamlit application directory
│   ├── app.py             # Main Streamlit application
│   └── model_utils.py     # Model loading and preprocessing utilities
├── requirements.txt        # Project dependencies
└── brain-tumor-detection-70-accuracy.py  # Training script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NeuroScan.AI.git
cd NeuroScan.AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

1. **Launch the Application**:
```bash
cd app
streamlit run app.py
```

2. **Upload MRI Scan**:
   - Click the "Upload MRI Scan" button
   - Select a clear, high-quality MRI image file
   - The system will automatically process and analyze the image

3. **Enter Patient Information**:
   - Fill in the patient details form, including:
     - Personal Information (name, age, gender)
     - Medical History
     - Current Symptoms
     - Previous Treatments
     - Family History
     - Lifestyle Factors

4. **Review Analysis**:
   - View the tumor classification results
   - Check confidence scores and probability distribution
   - Review AI-generated treatment recommendations

5. **Generate PDF Report**:
   - Click "Generate Report" to create a comprehensive medical report
   - The PDF report includes:
     - Patient information
     - MRI scan analysis
     - Classification results
     - Personalized treatment recommendations
     - Medical disclaimer
     - Doctor's approval section

6. **Medical Validation**:
   - Take the generated report to your healthcare provider
   - The report includes a dedicated section for doctor's review and approval
   - Your doctor can:
     - Review the AI analysis
     - Add additional notes
     - Sign and approve the recommendations
     - Add their medical license number for validation

## Important Notes

- 🏥 **Medical Disclaimer**: This tool is designed to assist medical professionals, not replace them. All recommendations should be reviewed and approved by a qualified healthcare provider.
- 📋 **Data Privacy**: Patient information is processed locally and not stored on any external servers.
- 🔒 **Security**: The system uses secure, encrypted connections for all API communications.

## Model Information

The system uses a fine-tuned MobileNetV2 architecture trained on a comprehensive dataset of brain MRI scans. The model achieves high accuracy in classifying different types of brain tumors while maintaining efficient processing times.

## Citations

This project builds upon and acknowledges the following sources:

1. Brain Tumor Classification Dataset by Sartaj Bhuvaji
   - Source: Kaggle
   - URL: [Brain Tumor Classification Dataset](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)

2. Model Development Reference
   - Author: Amirhosein Mousavian
   - Source: Kaggle
   - Accuracy: 70%
   - URL: [Brain Tumor Detection](https://www.kaggle.com/mousavian/brain-tumor-detection)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- [Anshul Ganumpally](https://github.com/anshulg614)

## Important Note

This application is for research purposes only and should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical diagnosis. 