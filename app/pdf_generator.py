from fpdf import FPDF
import datetime
import os

class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.add_page()
        # Set larger margins for better readability
        self.pdf.set_margins(20, 20, 20)
        self.pdf.set_auto_page_break(auto=True, margin=20)
        
    def add_header(self):
        """Add report header with logo and title."""
        self.pdf.set_font('Helvetica', 'B', 20)
        self.pdf.cell(0, 15, 'NueroScan.AI Report', ln=True, align='C')
        
        # Add timestamp
        self.pdf.set_font('Helvetica', '', 10)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pdf.cell(0, 5, f'Generated on: {current_time}', ln=True, align='R')
        self.pdf.ln(10)

    def add_patient_info(self, patient_data):
        """Add patient information section."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, 'Patient Information', ln=True)
        
        self.pdf.set_font('Helvetica', '', 12)
        for section, data in patient_data.items():
            if data:  # Only add sections that have data
                self.pdf.set_font('Helvetica', 'B', 12)
                self.pdf.cell(0, 8, section, ln=True)
                self.pdf.set_font('Helvetica', '', 12)
                
                for key, value in data.items():
                    if value:  # Only add fields that have values
                        self.pdf.cell(0, 8, f'{key}: {value}', ln=True)
                self.pdf.ln(2)

    def add_mri_image(self, image_path):
        """Add the MRI scan image."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, 'MRI Scan', ln=True)
        self.pdf.image(image_path, x=10, y=None, w=190)
        self.pdf.ln(5)

    def add_classification_results(self, predicted_class, confidence, all_probabilities):
        """Add tumor classification results."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, 'Classification Results', ln=True)
        self.pdf.ln(2)
        
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.cell(0, 8, f'Predicted Tumor Type: {predicted_class}', ln=True)
        confidence_fixed = confidence / 100 if confidence > 100 else confidence
        self.pdf.cell(0, 8, f'Confidence: {confidence_fixed:.2f}%', ln=True)
        
        self.pdf.ln(4)
        self.pdf.cell(0, 8, 'Detailed Probabilities:', ln=True)
        for tumor_type, prob in all_probabilities.items():
            prob_fixed = prob / 100 if prob > 100 else prob
            self.pdf.cell(0, 6, f'{tumor_type}: {prob_fixed:.2f}%', ln=True)
        
        self.pdf.ln(8)

    def add_treatment_recommendations(self, recommendations):
        """Add treatment recommendations with proper formatting."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, 'Treatment Recommendations', ln=True)
        self.pdf.ln(5)
        
        try:
            # Ensure recommendations is a string
            if isinstance(recommendations, list):
                recommendations = "\n".join(recommendations)
            elif not isinstance(recommendations, str):
                recommendations = str(recommendations)
            
            # Clean any markdown formatting and special characters
            clean_text = recommendations.replace('*', '').replace('_', '')
            # Replace any problematic Unicode characters with ASCII equivalents
            clean_text = clean_text.encode('ascii', 'replace').decode()
            
            # Split into sections
            sections = clean_text.split('\n')
            
            # Calculate effective page width
            effective_width = self.pdf.w - 40
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Check if it's a main section header (numbered sections)
                if any(section.startswith(f"{i}.") for i in range(1, 6)):
                    self.pdf.ln(8)
                    self.pdf.set_font('Helvetica', 'B', 13)
                    self.pdf.multi_cell(effective_width, 8, section)
                    self.pdf.ln(4)
                    self.pdf.set_font('Helvetica', '', 12)
                
                # Check if it's a subsection (starts with -)
                elif section.startswith('-'):
                    self.pdf.set_font('Helvetica', '', 12)
                    self.pdf.set_x(30)
                    indent_width = effective_width - 20
                    self.pdf.multi_cell(indent_width, 6, section)
                    self.pdf.ln(2)
                
                # Regular text
                else:
                    self.pdf.set_font('Helvetica', '', 12)
                    self.pdf.multi_cell(effective_width, 6, section)
                    self.pdf.ln(2)
            
            self.pdf.ln(8)
            
        except Exception as e:
            self.pdf.set_font('Helvetica', '', 12)
            basic_text = str(recommendations).encode('ascii', 'replace').decode()
            self.pdf.multi_cell(effective_width, 6, "Basic Recommendations:\n" + basic_text)
            self.pdf.ln(8)

    def add_disclaimer(self):
        """Add medical disclaimer."""
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.cell(0, 10, 'Medical Disclaimer', ln=True)
        
        disclaimer_text = """This report is generated by an AI system and is for informational purposes only. It should not be considered as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."""
        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.multi_cell(0, 8, disclaimer_text)

    def add_doctors_approval(self):
        """Add a section for doctor's approval."""
        self.pdf.add_page()
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, "Doctor's Approval", ln=True)
        
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.multi_cell(0, 8, """I have reviewed this AI-generated report and confirm that the information provided is accurate and consistent with my professional medical assessment.""")
        
        self.pdf.ln(10)
        
        # Add signature lines
        self.pdf.cell(80, 8, "Doctor's Name: _________________________", ln=True)
        self.pdf.ln(5)
        self.pdf.cell(80, 8, "Medical License #: _____________________", ln=True)
        self.pdf.ln(5)
        self.pdf.cell(80, 8, "Signature: ____________________________", ln=True)
        self.pdf.ln(5)
        self.pdf.cell(80, 8, "Date: ________________________________", ln=True)
        
        # Add notes section
        self.pdf.ln(10)
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.cell(0, 8, "Additional Notes:", ln=True)
        self.pdf.ln(5)
        
        # Add lines for notes
        self.pdf.set_font('Helvetica', '', 12)
        for _ in range(4):
            self.pdf.cell(0, 8, "_____________________________________________", ln=True)
            self.pdf.ln(5)

    def generate_report(self, output_path):
        """Save the PDF report."""
        try:
            # Add doctor's approval section before generating
            self.add_doctors_approval()
            self.pdf.output(output_path)
        except Exception as e:
            # If Times font fails, fallback to built-in font
            print(f"Warning: {str(e)}")
            print("Falling back to Helvetica font...")
            self.pdf = FPDF()
            self.pdf.set_font('Helvetica', '')
            self.generate_report(output_path) 