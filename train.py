from datetime import datetime
import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import csv

INPUT_SHAPE = (42,)
OUTPUT_SHAPE = 5

EPOCH_SIZE = 1000
BATCH_SIZE = 64


def create_model() -> keras.Sequential:
    """
    Basic model for training.
    :return: keras.Sequential
    """
    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Dense(21, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(21, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(10, activation='relu'),
            layers.Dense(OUTPUT_SHAPE, activation='softmax')
        ]
    )

    model.summary()
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data(dataset: str, columns: tuple):
    """
    Load data into arrays from csv
    :param dataset: path to dataset
    :param columns: train_x data columns
    :return:
    """

    x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(columns[0], columns[1])))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=42)

    return x_train, x_test, y_train, y_test


def main():
    x_train, x_test, y_train, y_test = load_data('data/gestures/merged.csv', (1, INPUT_SHAPE[0]+1))
    model = create_model()
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(x_train, y_train, epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])
    model.save('models/default.h5')


if __name__ == '__main__':
    main()
