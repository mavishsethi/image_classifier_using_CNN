# Cat vs Dogs Classifier

This is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images as either cats or dogs.

---

## Project Overview

- Dataset used: [Dogs vs Cats dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats) from Kaggle
- Images are resized to 256x256 pixels
- Model architecture consists of 3 convolutional layers with batch normalization and max pooling, followed by fully connected dense layers with dropout
- Binary classification using sigmoid activation in the output layer
- Trained for 10 epochs with Adam optimizer and binary cross-entropy loss

---

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV (`cv2`)
- Matplotlib

You can install the necessary packages with:

```bash
pip install tensorflow keras opencv-python matplotlib
