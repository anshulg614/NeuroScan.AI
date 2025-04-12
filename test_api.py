import google.generativeai as genai

# Configure the API key
API_KEY = 'AIzaSyD5Tl180d4wCc_3V0EQQrvpO1qK04HM-p0'
genai.configure(api_key=API_KEY)

def test_api():
    try:
        # Initialize the model
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
        # Create a simple prompt
        prompt = "Hello, can you tell me what 2+2 is?"
        
        # Generate a response
        response = model.generate_content(prompt)
        
        print("API Key is working!")
        print("Response:", response.text)
        return True
    except Exception as e:
        print("Error occurred:", str(e))
        return False

if __name__ == "__main__":
    test_api() 