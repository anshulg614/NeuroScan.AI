"""Usage guide and help text for the NeuroScan.AI application."""

def get_welcome_text():
    return """
    # Welcome to NeuroScan.AI 🧠

    NeuroScan.AI helps medical professionals analyze brain MRI scans and generate comprehensive medical reports with personalized treatment recommendations.
    """

def get_usage_steps():
    return [
        {
            "title": "1. Upload MRI Scan",
            "description": """
            - Use the upload button to select a brain MRI scan image
            - Supported formats: JPG, JPEG, PNG
            - Ensure the image is clear and properly oriented
            - The system will automatically process and analyze the image
            """
        },
        {
            "title": "2. Enter Patient Information",
            "description": """
            Fill in the patient details in the provided form sections:
            - Personal Information (name, age, gender)
            - Medical History
            - Current Symptoms
            - Previous Treatments
            - Family History
            - Lifestyle Factors
            
            This information helps generate personalized recommendations.
            """
        },
        {
            "title": "3. Review Analysis Results",
            "description": """
            The system will display:
            - Tumor classification result
            - Confidence score
            - Probability distribution for different tumor types
            - AI-generated treatment recommendations
            """
        },
        {
            "title": "4. Generate PDF Report",
            "description": """
            Click "Generate Report" to create a comprehensive medical report including:
            - Patient information
            - MRI scan analysis
            - Classification results
            - Personalized treatment recommendations
            - Medical disclaimer
            - Doctor's approval section
            """
        },
        {
            "title": "5. Medical Validation",
            "description": """
            - Share the generated report with your healthcare provider
            - The report includes a dedicated section for doctor's:
              • Review and approval
              • Additional notes
              • Signature
              • Medical license number
            - Follow the approved recommendations under medical supervision
            """
        }
    ]

def get_disclaimer():
    return """
    ⚠️ **Medical Disclaimer**
    
    This tool is designed to assist medical professionals, not replace them. All recommendations should be reviewed and approved by a qualified healthcare provider before implementation. The system:
    - Processes data locally for privacy
    - Uses secure connections for API communications
    - Does not store patient information
    - Requires medical validation for all recommendations
    """

def get_help_text():
    return """
    ## Need Help?
    
    If you encounter any issues:
    1. Check that your image is in a supported format
    2. Ensure all required patient information is filled
    3. Verify your internet connection for AI recommendations
    4. Make sure the PDF report downloads completely
    
    For technical support or feature requests, please visit our GitHub repository.
    """ 