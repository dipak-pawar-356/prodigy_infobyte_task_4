import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to load and preprocess the dataset
def load_dataset():
    # Load images and labels
    # For simplicity, let's assume you have images of hand gestures and corresponding labels
    # Ensure your dataset is balanced across different classes
    # For example, for three classes: fist (class 0), open palm (class 1), peace sign (class 2)
    # X_train, y_train = load_images_and_labels('path_to_training_data')
    # X_test, y_test = load_images_and_labels('path_to_testing_data')

    # For demonstration purposes, let's create placeholder data
    X_train = np.random.rand(300, 100, 100, 3)  # 300 images of size 100x100 with 3 channels (RGB)
    y_train = np.random.randint(0, 3, size=(300,))  # Random labels
    X_test = np.random.rand(100, 100, 100, 3)
    y_test = np.random.randint(0, 3, size=(100,))

    # Normalize pixel values to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    return X_train, y_train, X_test, y_test

# Define the CNN model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Load dataset
X_train, y_train, X_test, y_test = load_dataset()

# Split dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define model parameters
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

# Create and compile the model
model = create_model(input_shape, num_classes)

# Train the model
train_model(model, X_train, y_train, X_val, y_val)

# Save the model for later use
model.save('hand_gesture_model.h5')
