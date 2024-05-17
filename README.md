# EOG Signal Classification Project
![image](https://github.com/Shehab611/HCI_Project/assets/91502893/251a0539-9a30-4617-8324-8561b8c5e03e)


## Overview

This project aims to classify Electrooculography (EOG) signals using various machine learning algorithms. EOG signals are typically used to measure eye movements and can provide valuable insights into various physiological and neurological conditions. The project involves preprocessing the EOG signals, extracting statistical features, and applying multiple classifiers to achieve optimal performance.

## Preprocessing

The EOG signals are preprocessed to filter out noise and artifacts. The key steps include:
1. **Band-pass Filtering**: Applying a band-pass filter to retain frequencies within the range of 0.5 to 20 Hz.
2. **Mean Subtraction**: Removing the mean to center the signals around zero.
3. **Standardization**: Scaling the signals to have zero mean and unit variance.

## Feature Extraction

After preprocessing, statistical features are extracted from the EOG signals:
1. **Statistical Features**:
   - **Standard Deviation**: Measures the dispersion or variability of the data points around the mean.
   - **Variance**: Quantifies the degree of dispersion of the data points.
2. **Wavelet Coefficients**: Using wavelet decomposition with a 'db4' wavelet and 2 levels of decomposition. The A2 coefficients are extracted to capture the relevant frequency range (0.5 to 20 Hz).

These features are critical for capturing the underlying patterns and variability in the EOG signals, aiding in the classification process.

## Machine Learning Models

Several classifiers are used to classify the EOG signals. The models and their respective parameters are as follows:

1. **Logistic Regression**:
   - `max_iter=50`
   - `random_state=42`
   - `C=0.1`
   - `penalty='l2'`
   - `solver='newton-cg'`

2. **Decision Tree Classifier**:
   - `max_features='sqrt'`
   - `max_depth=15`
   - `min_samples_leaf=2`
   - `min_samples_split=10`

3. **Random Forest Classifier**:
   - `max_depth=5`
   - `min_samples_leaf=1`
   - `min_samples_split=5`
   - `n_estimators=50`
   - `max_features='log2'`

4. **Support Vector Classifier (Linear)**:
   - `kernel='linear'`
   - `C=10`
   - `degree=2`
   - `gamma='scale'`

5. **Support Vector Classifier (RBF)**:
   - `kernel='rbf'`
   - `C=10`
   - `gamma=0.001`

6. **Gaussian Naive Bayes**:
   - No parameters specified.

7. **AdaBoost Classifier**:
   - `learning_rate=0.01`
   - `n_estimators=50`
   - `algorithm='SAMME.R'`

8. **Gradient Boosting Classifier**:
   - `learning_rate=0.01`
   - `max_depth=3`
   - `n_estimators=50`
   - `max_features='sqrt'`
   - `min_samples_leaf=2`
   - `min_samples_split=5`

## Evaluation

The performance of each model is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. The results are saved in the `results/` directory for further analysis.

## Installation

To set up the project, follow these steps:

Clone the repository:
   ```bash
   git clone https://github.com/MernaAbdallah/eog-signal-classification.git
   cd eog-signal-classification
   ```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.
