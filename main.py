import math
from math import sqrt
import cv2
import numpy as np
from utils import HandTracker, detect_qr_code
import pandas as pd
import csv
from tensorflow.keras.models import load_model, Sequential


def train(cap: cv2.VideoCapture, tracker: HandTracker):
    append = True
    recording = False
    label = 0
    dataset = list()

    while True:
        _, frame = cap.read()
        tracker.hands_finder(frame, False)
        landmarks = tracker.position_finder(frame, normalized=False)

        if len(landmarks) > 0:
            for lm in landmarks:
                cv2.circle(frame, lm, radius=2, thickness=1, color=(0, 255, 0))

            cv2.putText(frame, f'Label: {label}', (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 0))

            landmarks = np.array(tracker.position_finder(frame, normalized=True)).flatten()
            landmarks = np.insert(landmarks, 0, label, axis=0)
            cv2.putText(frame, f'Recording: {recording}', (0, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))
            if recording:
                if append:
                    with open('data/data.csv', 'a', newline='') as file:
                        write = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        write.writerow(landmarks)
                else:
                    dataset.append(landmarks)

        cv2.imshow('main', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('x'):
            recording = not recording
        elif 48 <= key <= 57:
            label = chr(key)
    if not append:
        with open('data/data.csv', 'w', newline='') as file:
            write = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            write.writerows(dataset)


def predict_gesture(cap: cv2.VideoCapture, tracker: HandTracker):
    model: Sequential = load_model('./models/4_model.h5')
    classes = ['down', 'stop', 'left', 'right', 'up', 'down', 'pinch']

    while True:
        _, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 50

        tracker.hands_finder(frame)
        lms = tracker.position_finder(frame, normalized=True)

        if lms:
            landmark = np.array(lms).reshape((42,))

            predictions = model.predict(np.array([landmark]))
            predicted = np.argmax(np.squeeze(predictions))
            if np.squeeze(predictions)[predicted] > 0:
                predicted_class = classes[predicted]
                cv2.putText(frame, predicted_class, (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 0))
                for i, prediction in enumerate(np.squeeze(predictions)):
                    cv2.putText(frame, '{0:.2f}%'.format(prediction * 100), org=(0, 20 + int(height - (i * 10))),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.25)
                    cv2.rectangle(frame, (50, 15 + int(height - (i * 10))),
                                  (50 + int((width - 50) * prediction), 20 + int(height - (i * 10))),
                                  color=(int(255 * prediction), int(255 * prediction), int(255 * prediction)),
                                  thickness=-1)
                    cv2.putText(frame, f'{classes[i]}', org=(50 + int((width - 50) * prediction), 20 + int(height - (i * 10))),
                                fontScale=0.25, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
                if predicted_class == 'pinch':
                    distance = sqrt((lms[8][0] - lms[4][0]) ** 2 + (lms[8][1] - lms[4][1]) ** 2)
                    # print(distance)
                    cv2.line(frame, (int(lms[8][0] * frame.shape[1]), int(lms[8][1] * frame.shape[0])),
                             (int(lms[4][0] * frame.shape[1]), int(lms[4][1] * frame.shape[0])),
                             color=(255, 0, 255), thickness=2)
        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def projection(cap: cv2.VideoCapture, tracker: HandTracker):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sharpen = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    while True:
        _, frame = cap.read()
        high: list = [0, 0]
        low: list = [width, height]

        tracker.hands_finder(frame, draw=False)

        landmarks = tracker.position_finder(frame, hand_no=0)

        for lm in landmarks:
            if lm[0] < low[0]:
                low[0] = lm[0]
            if lm[1] < low[1]:
                low[1] = lm[1]
            if lm[0] > high[0]:
                high[0] = lm[0]
            if lm[1] > high[1]:
                high[1] = lm[1]
            frame = cv2.arrowedLine(frame, landmarks[0], lm,
                                    color=(0, 0, math.dist(landmarks[0], lm)), thickness=1, tipLength=0.02)
            cv2.circle(frame, lm, color=(0, 0, 255), radius=2)

        cv2.rectangle(frame, low, high, color=(0, 0, 255))

        pts1 = np.float32([low, [high[0], low[0]], [low[0], high[1]], high])
        pts2 = np.float32([[0, 0], [high[0] - low[0], 0], [0, high[1] - low[1]], [high[0] - low[0], high[1] - low[1]]])
        # pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

        # matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # result = cv2.warpPerspective(frame, matrix, (400, 600))
        cv2.imshow('frame', frame)
        # cv2.imshow('projection', result)
        if ((high[0] - low[0]) > 0) and ((high[1] - low[1]) > 0):
            cropped = frame[low[1]:high[1], low[0]:high[0]]
            cropped = cv2.resize(cropped, (400, 400))
            cropped = cv2.filter2D(cropped, -1, sharpen)
            cropped = tracker.hands_finder(cropped, draw=True)
            cv2.imshow('cropped', cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    # train(cap, tracker)
    predict_gesture(cap, tracker)
    # projection(cap, tracker)
    """
    database = list()
    with open('data/data.csv') as file:
        csv_file = csv.reader(file)

        for row in csv_file:
            temp = list()
            temp.append(row[0])
            temp.extend([float(row[1]) - float(row[3]), float(row[2]) - float(row[4])])
            for i in range(5, len(row), 2):
                temp.append(math.dist([float(row[1]), float(row[2])], [float(row[i]), float(row[i+1])]))
                print(i)
            database.append(temp)

    with open('data/data_dist.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(database)

    """
    """
    database = list()

    with open('data/data.csv') as file:
        csv_file = csv.reader(file)

        for row in csv_file:
            temp = list()
            temp.append(row[0])
            for i in range(3, len(row), 2):
                x = row[i] * -1 if row[1] > row[i] else row[i]
                y = row[i+1] * -1 if row[2] > row[i+1] else row[i+1]
                temp.extend([x, y])
            database.append(temp)
    with open('data/data_dist_signed.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(database)
    """


def landmark_to_distance(lms:list):
    result = list()

    result.extend([float(lms[0]) - float(lms[2]), float(lms[1]) - float(lms[3])])
    for i in range(4, len(lms), 2):
        result.append(math.dist([float(lms[0]), float(lms[1])], [float(lms[i]), float(lms[i+1])]))
    return result


if __name__ == '__main__':
    main()

# 0 = stop, 1 = pinch, 2 = up, 3 = down, 4 = left, 5 = right, 6 = peace
