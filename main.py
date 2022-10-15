import cv2
import numpy as np
from utils import detectQRCode, GestureControl


def main():
    classes = ['okay', 'peace', 'thumbs up', 'thumbs down',
               'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
    cap = cv2.VideoCapture(0)
    gesture = GestureControl("mp_hand_gesture", classes)
    while True:
        _, frame = cap.read()

        prediction = gesture.predict(frame)
        print(prediction)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
