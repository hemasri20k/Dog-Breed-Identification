import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing import image
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# Load the labels.csv file
 # Function to preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match the input size expected by the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess image for MobileNetV2
    return img
# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')
# Function to predict breed of a single dog image
def predict_dog_breed(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    decoded_predictions = decode_predictions(prediction, top=1)[0]
    _, breed, confidence = decoded_predictions[0]
    return breed, confidence
# Function to display dog image and predicted breed
def display_dog_image_with_breed(image_path, breed):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    plt.imshow(img)
    plt.title('Predicted Breed: ' + breed)
    plt.axis('off')
    plt.show()
# Example usage:
image_path = 'husky.jpeg'  # Specify the path to the input dog image
breed, confidence = predict_dog_breed(image_path)
print("Predicted Breed:", breed)
print("Confidence:", confidence)
# Evaluate accuracy using the same image
def evaluate_accuracy_single_image(model, image_path, ground_truth):
    breed, confidence = predict_dog_breed(image_path)
    filename = os.path.basename(image_path)
    if ground_truth.get(filename) == breed:
        return 1
    else:
        return 0

# Ground truth label for the example image
ground_truth_label = breed

# Evaluate accuracy on the example image
accuracy_single_image = evaluate_accuracy_single_image(model, image_path, {image_path: ground_truth_label})
print("Accuracy:", accuracy_single_image)
display_dog_image_with_breed(image_path, breed)



