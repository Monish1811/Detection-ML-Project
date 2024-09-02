import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_and_preprocess_images(folder_path):
    images = []
    labels = []

    for class_label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_label)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                # Resize the image to a fixed size (e.g., 64x64 pixels)
                image = cv2.resize(image, (64, 64))
                # Normalize pixel values to the range [0, 1]
                image = image / 255.0
                # Flatten the image to a 1D array
                flattened_image = image.flatten()
                images.append(flattened_image)
                labels.append(class_label)

    return np.array(images), np.array(labels)

# Set the path to your image data folder
image_folder = '/content/drive/MyDrive/Datasets/Retina'

# Loop over 20 epochs
epochs = 20
accuracy_history = []

for epoch in range(epochs):
    # Load and preprocess images
    X, y = load_and_preprocess_images(image_folder)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize the XGBoost classifier with tuned hyperparameters
    xgb_classifier = XGBClassifier(
        learning_rate=0.1,  # Adjust as needed
        n_estimators=100,   # Adjust as needed
        max_depth=6 + epoch,  # Increase max_depth with each epoch
        subsample=0.8,      # Adjust as needed
        colsample_bytree=0.8,  # Adjust as needed
        random_state=42
    )

    # Train the classifier
    xgb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = xgb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracy_history.append(accuracy)
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%')

# Plotting the accuracy over epochs
plt.plot(range(1, epoch + 2), accuracy_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('XGBoost Accuracy over Epochs')
plt.grid(True)
plt.show()
