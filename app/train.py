import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import os

class BrainTumorModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3,3), activation='relu', padding='same', input_shape=self.input_shape),
            AveragePooling2D(pool_size=(2,2)),
            BatchNormalization(),

            # Second Convolutional Block
            Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
            MaxPooling2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.2),

            # Flatten layer
            Flatten(),

            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        return model

def train_model():
    # Data generators
    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train = train_gen.flow_from_directory(
        "Brain-Tumor-Classification-DataSet/Training",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    # Load validation data
    val = train_gen.flow_from_directory(
        "Brain-Tumor-Classification-DataSet/Training",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    # Load test data
    test = test_gen.flow_from_directory(
        "Brain-Tumor-Classification-DataSet/Testing",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=True
    )

    # Create and compile model
    model = BrainTumorModel().build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train,
        validation_data=val,
        epochs=25,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test)
    print(f"\nTest accuracy: {test_accuracy:.2f}")

    # Save model
    model.save('brain_tumor_model.h5')
    print("\nModel saved as brain_tumor_model.h5")

if __name__ == "__main__":
    train_model() 