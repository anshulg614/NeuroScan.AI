def get_recommendations(tumor_type, confidence):
    """Generate treatment recommendations based on tumor type."""
    recommendations = []
    
    if tumor_type == "glioma":
        recommendations.extend([
            "Primary Treatment Options:",
            "1. Surgical resection to remove as much of the tumor as safely possible",
            "2. Radiation therapy following surgery",
            "3. Chemotherapy with temozolomide",
            
            "Additional Considerations:",
            "- Regular MRI monitoring every 2-3 months",
            "- Anti-seizure medications if needed",
            "- Rehabilitation therapy as required"
        ])
    
    elif tumor_type == "meningioma":
        recommendations.extend([
            "Primary Treatment Options:",
            "1. Observation with regular MRI scans for slow-growing tumors",
            "2. Surgical removal for symptomatic or growing tumors",
            "3. Radiation therapy for residual tumor or inoperable cases",
            
            "Additional Considerations:",
            "- Annual MRI monitoring if observation approach",
            "- Anti-seizure medications if needed",
            "- Regular neurological assessments"
        ])
    
    elif tumor_type == "pituitary":
        recommendations.extend([
            "Primary Treatment Options:",
            "1. Medication therapy for hormone-secreting tumors",
            "2. Transsphenoidal surgery for larger tumors",
            "3. Radiation therapy for residual tumor",
            
            "Additional Considerations:",
            "- Regular hormone level monitoring",
            "- Vision and field tests",
            "- Replacement hormone therapy if needed"
        ])
    
    else:  # no_tumor
        recommendations.extend([
            "Recommendations:",
            "1. Regular health check-ups",
            "2. Follow-up MRI scan in 6-12 months",
            "3. Report any new neurological symptoms promptly",
            
            "Additional Considerations:",
            "- Maintain healthy lifestyle",
            "- Monitor any changes in symptoms",
            "- Regular medical check-ups"
        ])
    
    if confidence < 85:
        recommendations.append(
            "Note: Due to confidence level below 85%, additional diagnostic tests and expert consultation are strongly recommended."
        )
    
    return recommendations 