import tensorflow as tf
import os

# Disable GPU and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
import numpy as np
from PIL import Image
import cv2

def load_model():
    """Load the trained MobileNetV2 model with saved weights."""
    # Create base MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    # Get the absolute path to the weights file
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_weights.weights.h5')
    
    # Load the weights
    model.load_weights(weights_path)
    
    return model

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    # Resize image to 224x224
    image = image.resize((224, 224))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def get_class_names():
    """Return the class names in the correct order."""
    return ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor'] 