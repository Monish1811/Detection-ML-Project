import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_images(folder_path):
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
                # Flatten the image to a 1D array
                flattened_image = image.flatten()
                images.append(flattened_image)
                labels.append(class_label)

    return np.array(images), np.array(labels)

# Set the path to your image data folder
image_folder = '/content/drive/MyDrive/Datasets/Retina'

# Initialize lists to store accuracy and epoch values for graphing
epoch_values = []
accuracy_values = []

# Loop over 20 epochs
for epoch in range(20):
    # Load and shuffle images for each epoch
    X, y = load_images(image_folder)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set different values for C (regularization parameter) to iterate over
    C_values = [1.0]

    # Loop over different values of C
    for C in C_values:
        # Initialize the SVM classifier
        svm_classifier = SVC(C=C)

        # Train the classifier
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = svm_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Append values for graphing
        epoch_values.append(epoch + 1)
        accuracy_values.append(accuracy * 100)

        print(f'Epoch {epoch + 1}, Accuracy with C={C}: {accuracy*100:.2f}%')

# Plotting the accuracy graph
plt.plot(epoch_values, accuracy_values, marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()
