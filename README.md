# NueroScan.AI

A deep learning-powered brain tumor classification system using MobileNetV2 architecture. This application can classify brain MRI scans into four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor.

## Features

- Real-time brain tumor classification from MRI scans
- User-friendly web interface built with Streamlit
- Powered by MobileNetV2 architecture (71% accuracy)
- Fast and efficient image processing

## Tech Stack

- TensorFlow 2.15.0 (Deep Learning Framework)
- Streamlit 1.32.0 (Web Application Framework)
- OpenCV 4.9.0.80 (Image Processing)
- Pillow 10.2.0 (Image Handling)
- NumPy 1.24.3 (Numerical Operations)

## Project Structure

```
NueroScan.AI/
├── app/                    # Streamlit application directory
│   ├── app.py             # Main Streamlit application
│   └── model_utils.py     # Model loading and preprocessing utilities
├── requirements.txt        # Project dependencies
└── brain-tumor-detection-70-accuracy.py  # Training script
```

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:anshulg614/NueroScan.AI.git
   cd NueroScan.AI
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

## Usage

1. Train the model (optional, pre-trained weights provided):
   ```bash
   python brain-tumor-detection-70-accuracy.py
   ```

2. Run the Streamlit application:
   ```bash
   cd app
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## Model Architecture

The system uses a MobileNetV2 base model (pre-trained on ImageNet) with additional custom layers:
- Global Average Pooling
- Dense layer (128 units, ReLU activation)
- Dropout layer (0.5)
- Output layer (4 units, Softmax activation)

## License

[MIT License](LICENSE)

## Contributors

- [Anshul Ganumpally](https://github.com/anshulg614)

## Important Note

This application is for research purposes only and should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical diagnosis. 