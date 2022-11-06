import os
import cv2 as cv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from deprecated.preprocess import prep_image

INPUT_SHAPE = (640, 360, 1)
NUM_OUTPUT = 2

EPOCH_SIZE = 15
BATCH_SIZE = 4


def create_model() -> keras.Sequential:
    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Dense(128),
            layers.Dense(64),
            layers.Dense(32),
            layers.Dense(16),
            layers.Dense(2),
            layers.Flatten(),
            layers.Dense(2, activation='sigmoid')
        ]
    )

    model.summary()
    model.compile(
        'adam',
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    return model


def load_data() -> tuple:
    train_x, train_y = [], []

    for file in os.listdir('../data/left')[0:10]:
        image = cv.imread(f'./data/left/{file}')
        image = prep_image(image)
        image = cv.cvtColor(src=image, dst=image, code=cv.COLOR_RGB2GRAY)
        image = np.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image = np.asarray(image, dtype='float32')
        train_x.append(image / 255.0)
        train_y.append(0)

    for file in os.listdir('../data/right')[0:10]:
        image = cv.imread(f'./data/right/{file}')
        image = prep_image(image)
        image = cv.cvtColor(src=image, dst=image, code=cv.COLOR_RGB2GRAY)
        image = np.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image = np.asarray(image, dtype='float32')
        train_x.append(image / 255.0)
        train_y.append(1)


    train_x = np.expand_dims(train_x, -1)
    train_y = np.array(train_y, dtype='float32')

    return train_x, train_y


def main():
    train_x, train_y = load_data()
    print(train_x.shape, train_y.shape)
    model = create_model()
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, validation_split=0.2)
    model.save('./model.h5')


if __name__ == '__main__':
    main()
