# Thyroid Cancer Classification

Thyroid cancer is a disease that is characterised by the uncontrolled growth of abnormal cells within the thyroid gland. While ultrasound imaging serves as the primary diagnostic tool for thyroid cancer, its interpretation can be subjective, leading to inter-observer variability. This project aims to address this challenge by developing a machine learning model that classifies thyroid ultrasound images into benign and malignant tumors using the Support Vector Machine (SVM) algorithm.

## Dataset

The dataset is sourced from Kaggle and 50 images are chosen for training, including both benign and malignant cases.


## Methodology

1. Image preprocessing:
- Ultrasound images underwent preprocessing techniques to enhance their quality and reduce noise. This included resizing, cropping, and median filtering.

2. Feature Extraction
Two types of features were extracted from the preprocessed images:
- Statistical Features: These features capture the overall intensity distribution of the image, including mean, median, and standard deviation.

- Texture Features: These features capture the local patterns and textures within the image, using Local Binary Patterns (LBP).

3. Classification
- The Support Vector Machine (SVM) algorithm was used to classify the extracted features into benign or malignant categories. 

## Results

- The trained SVM model achieved an overall accuracy of 90% on the test dataset.
- Benign Tumors (Class 0): Recall: 100%, Precision: 75%
- Malignant Tumors (Class 1): Recall: 86%, Precision: 100%

These results indicate that the model effectively identified all benign cases in the dataset while demonstrating excellent precision for malignant tumors.


## Acknowledgments

- This project utilizes the [DDTI Thyroid Ultrasound Images dataset](https://www.kaggle.com/dasmehdixtr/ddti-thyroid-ultrasound-images?select=100.xml) from Kaggle.
