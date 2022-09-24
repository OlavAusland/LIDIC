import cv2 as cv
import numpy as np
from tensorflow import keras
from preprocess import prep_image

def main():
    model = keras.models.load_model('./model.h5')

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()
        image = prep_image(frame)
        image = cv.cvtColor(src=image, dst=image, code=cv.COLOR_RGB2GRAY)
        image = np.expand_dims(image, axis=0)
        image = np.asarray(image, dtype='float32')
        image = image / 255.0
        image = np.reshape(image, (640, 360, 1))
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        print(prediction)
        cv.imshow('feed', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    main()
