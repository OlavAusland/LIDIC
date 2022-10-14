import cv2
import numpy as np
from utils import detectQRCode


def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        frame = detectQRCode(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
