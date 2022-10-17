import cv2
import numpy as np
from utils import HandTracker, detect_qr_code
import pandas as pd
import csv
from tensorflow.keras.models import load_model, Sequential


def train(cap: cv2.VideoCapture, tracker: HandTracker):
    label = 0
    dataset = list()

    while True:
        _, frame = cap.read()
        tracker.hands_finder(frame, False)
        landmarks = tracker.position_finder(frame, normalized=False)

        if len(landmarks) > 0:
            for lm in landmarks:
                cv2.circle(frame, lm, radius=2, thickness=1, color=(0, 255, 0))

            cv2.putText(frame, f'Label: {label}', (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))

            landmarks = np.array(tracker.position_finder(frame, normalized=True)).flatten()
            landmarks = np.insert(landmarks, 0, label, axis=0)
            dataset.append(landmarks)

        cv2.imshow('main', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif 48 <= key <= 57:
            label = chr(key)
    with open('data/gesture_data.csv', 'w', newline='') as file:
        write = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        write.writerows(dataset)


def predict_gesture(cap: cv2.VideoCapture, tracker: HandTracker):
    model: Sequential = load_model('models/4_model.h5')

    while True:
        _, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 50

        tracker.hands_finder(frame)
        landmark = tracker.position_finder(frame, normalized=True)

        if landmark:
            landmark = np.array(landmark).reshape((42,))
            predictions = model.predict(np.array([landmark]))

            cv2.putText(frame, str(np.argmax(np.squeeze(predictions))), (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 0))
            for i, prediction in enumerate(np.squeeze(predictions)):
                cv2.putText(frame, '{0:.2f}%'.format(prediction * 100), org=(0, 20 + int(height - (i * 10))),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.25)
                cv2.rectangle(frame, (50, 15 + int(height - (i * 10))), (50 + int((width-50) * prediction), 20 + int(height - (i * 10))),
                         color=(int(255 * prediction), int(255 * prediction), int(255 * prediction)), thickness=-1)

        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    # train(cap, tracker)
    predict_gesture(cap, tracker)


if __name__ == '__main__':
    main()
