# cat_vs_dogs_classifier.py

import os
import zipfile
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import cv2
import numpy as np

def unzip_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)  # normalize images to [0,1]
    return image, label

def load_datasets(train_dir, val_dir, batch_size=32, img_size=(256,256)):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size
    )
    val_ds = keras.utils.image_dataset_from_directory(
        directory=val_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size
    )
    train_ds = train_ds.map(process)
    val_ds = val_ds.map(process)
    return train_ds, val_ds

def create_model(input_shape=(256,256,3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.plot(history.history['accuracy'], color='red', label='train accuracy')
    plt.plot(history.history['val_accuracy'], color='blue', label='validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], color='red', label='train loss')
    plt.plot(history.history['val_loss'], color='blue', label='validation loss')
    plt.legend()
    plt.show()

def predict_image(model, image_path, img_size=(256,256)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, img_size)
    img = img / 255.0  # normalize like training images
    img = img.reshape((1, img_size[0], img_size[1], 3))
    prediction = model.predict(img)
    print(f"Prediction (probability of class 1): {prediction[0][0]:.4f}")
    return prediction

def main():
    # Assuming you have downloaded and extracted the dataset locally.
    # Change these paths to your dataset locations.
    train_dir = './train'
    val_dir = './test'
    
    # If you have zipped dataset and want to extract, uncomment:
    # unzip_data('dogs-vs-cats.zip', '.')

    print("Loading datasets...")
    train_ds, val_ds = load_datasets(train_dir, val_dir)

    print("Creating model...")
    model = create_model()

    model.summary()

    print("Training model...")
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)

    print("Plotting training history...")
    plot_history(history)

    # Predict on a test image
    test_image_path = './cat.jpeg'  # Change to your test image path
    print(f"Predicting on image: {test_image_path}")
    prediction = predict_image(model, test_image_path)

if __name__ == "__main__":
    main()
