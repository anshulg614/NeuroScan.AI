import google.generativeai as genai
import os

# Configure the API key
genai.configure(api_key='AIzaSyD5Tl180d4wCc_3V0EQQrvpO1qK04HM-p0')

def format_patient_info(patient_data):
    """Format patient information for the prompt."""
    sections = []
    
    for section, data in patient_data.items():
        if data:  # Only include sections with data
            section_info = [f"\n{section}:"]
            for key, value in data.items():
                if value:  # Only include fields with values
                    section_info.append(f"- {key}: {value}")
            sections.append("\n".join(section_info))
    
    return "\n".join(sections)

def get_treatment_recommendations(tumor_type, patient_data):
    """Generate treatment recommendations using Google AI Studio."""
    
    # Format patient information
    patient_info = format_patient_info(patient_data)
    
    # Create the prompt template
    prompt = f"""
    As a medical AI assistant specializing in neuro-oncology, provide comprehensive treatment recommendations for a brain tumor patient with the following details:

    DIAGNOSIS:
    - Tumor Type: {tumor_type}

    PATIENT INFORMATION:{patient_info}

    Please provide a detailed, personalized treatment plan including:

    1. TREATMENT RECOMMENDATIONS:
       - Primary treatment options (surgical and non-surgical)
       - Alternative treatment approaches
       - Treatment timeline and phases
       - Specific considerations based on patient's medical history and current condition

    2. RISK ASSESSMENT AND MANAGEMENT:
       - Potential complications and side effects
       - Risk factors based on patient profile
       - Contraindications with current medications
       - Monitoring requirements

    3. PROGNOSIS AND SUCCESS RATES:
       - Expected outcomes
       - Success rates for recommended treatments
       - Quality of life considerations
       - Recovery timeline

    4. FOLLOW-UP CARE PLAN:
       - Follow-up schedule
       - Monitoring protocols
       - Required tests and screenings
       - Signs/symptoms to watch for

    5. LIFESTYLE RECOMMENDATIONS:
       - Dietary modifications
       - Physical activity guidelines
       - Stress management
       - Daily life adjustments

    6. SUPPORT RESOURCES:
       - Support groups
       - Educational resources
       - Professional support services
       - Family support recommendations

    Format the response in clear sections with bullet points where appropriate. Consider all provided patient information in your recommendations.
    """
    
    # Generate the response
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    
    return response.text

def format_recommendations(recommendations):
    """Format the recommendations for display in Streamlit."""
    sections = recommendations.split('\n\n')
    formatted_output = []
    
    for section in sections:
        if section.strip():
            formatted_output.append(section)
    
    return formatted_output 