import os
import cv2
import numpy as np
from skimage import io, measure, feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path, new_width, new_height):
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (new_width, new_height))
    right_pixels_to_remove = 30
    left_pixels_to_remove = 28
    bottom_pixels_to_remove = 23
    cropped_image = resized_image[:, left_pixels_to_remove:-right_pixels_to_remove, :]
    cropped_image = cropped_image[:-bottom_pixels_to_remove, :]
    median_filtered_image = cv2.medianBlur(cropped_image, 3)
    return median_filtered_image

def calculate_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_features(sample):
    image_path = sample["image_path"]
    final_image = preprocess_image(image_path, new_width, new_height)
    lbp_features = calculate_lbp_features(final_image)
    labeled_image = measure.label(final_image > 0.5)
    regions = measure.regionprops(labeled_image)
    mean_intensity = np.mean(final_image)
    median_intensity = np.median(final_image)
    std_intensity = np.std(final_image)

    # Combine all features into a single array
    features = np.concatenate([lbp_features, [mean_intensity, median_intensity, std_intensity]])
    return features

def load_images(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename) 
            data.append({"image_path": img_path})

    return data

# Set the paths to the benign and malignant folders
benign_folder = r"C:\Thyroid_Cancer_Classification\Data\Benign"
malignant_folder = r"C:\Thyroid_Cancer_Classification\Data\Malignant"

# Load images from the folders
benign_data = load_images(benign_folder)
malignant_data = load_images(malignant_folder)

# New width and height for image resizing
new_width = 256
new_height = 256

# Create a list to store features and labels
X = []
y = []

# Organizing data and labels
for sample in benign_data:
    features = extract_features(sample)
    X.append(features)
    y.append(0)  # Label 0 for benign

for sample in malignant_data:
    features = extract_features(sample)
    X.append(features)
    y.append(1)  # Label 1 for malignant

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the SVM classifier
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
