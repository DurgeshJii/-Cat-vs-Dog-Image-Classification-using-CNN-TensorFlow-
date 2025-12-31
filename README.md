# Cat-vs-Dog-Image-Classification-using-CNN-TensorFlow-
This project demonstrates an end-to-end image classification pipeline using Convolutional Neural Networks (CNN) to classify images as Cat or Dog.  The model is built from scratch using TensorFlow &amp; Keras, trained on preprocessed image data stored in CSV format, and evaluated on a separate test dataset.

# 📌 Project Overview

Implemented a binary image classification model

Used CNN architecture for feature extraction

Trained and tested on RGB images (100×100 resolution)

Achieved ~78% training accuracy and ~66% test accuracy

# 🧠 Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

CNN (Deep Learning)

# 🗂 Dataset Details

Training Data:

input.csv → Image pixel values

labels.csv → Class labels (Cat / Dog)

Testing Data:

input_test.csv

labels_test.csv

Images reshaped to: (100, 100, 3)

Normalized using pixel / 255.0

# 🏗 Model Architecture

Conv2D + ReLU

MaxPooling

Conv2D + ReLU

MaxPooling

Flatten

Dense (64 units)

# Output Layer (Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

# 📊 Results

Training Accuracy: ~78%

Test Accuracy: ~66%

Random image prediction with visualization

# 📷 Sample Output

The model predicts whether the given image is a Cat or Dog and displays the image using Matplotlib.

# 🚀 Key Learnings

CNN fundamentals and image preprocessing

Model training & evaluation

Overfitting awareness

Real-world ML workflow

# 📌 Future Improvements

Use real image datasets instead of CSVs

Data augmentation

Transfer learning (VGG16 / ResNet)

Hyperparameter tuning

# Durgesh Yadav
Aspiring Data Analyst | ML Enthusiast | AI Learner
