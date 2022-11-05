import math
from math import sqrt
from typing import List

import cv2
import numpy as np
from utils import HandTracker, detect_qr_code
import pandas as pd
import csv
from tensorflow.keras.models import load_model, Sequential


def create_dataset(cap: cv2.VideoCapture, tracker: HandTracker):
    """
    Will capture hand datapoints and save it to a csv file.

    :param cap: Capture device
    :param tracker: HandTracker object to detect the hand
    :return: Nothing
    """
    append = True
    recording = False  # If true it will add the points to the database
    label = 0  # Choose which label will be written with the frame
    dataset = list()  # Points and label will be writen to this file as a 1d array

    while True:
        # read frame
        _, frame = cap.read()
        tracker.hands_finder(frame, False)

        # get the landmarks from each joint as an array
        landmarks = tracker.position_finder(frame, normalized=False)

        # check if a hand is detected
        if len(landmarks) > 0:
            # foreach element in landmarks draw a circle at the position
            for lm in landmarks:
                cv2.circle(frame, lm, radius=2, thickness=1, color=(0, 255, 0))

            # write the current label being recorded to the window
            cv2.putText(frame, f'Label: {label}', (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 0))

            # get the points of the hand as normalized values and flatten the array to a 1d array, this might be
            # will fix later :)
            landmarks = np.array(tracker.position_finder(frame, normalized=True)).flatten()
            # add label to the first position - important for the training part!
            landmarks = np.insert(landmarks, 0, label, axis=0)
            # write if you are recording to the screen
            cv2.putText(frame, f'Recording: {recording}', (0, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))
            if recording:
                # if append is true append ot the csv file, if not overwrite the file
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
    # overwrite the data.csv file
    if not append:
        with open('data/data.csv', 'w', newline='') as file:
            write = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            write.writerows(dataset)


def predict_gesture(cap: cv2.VideoCapture, tracker: HandTracker, model_path: str, classes: List[str]):
    """
    Predict gestures from a live feed using a capture device (default webcam).
    :param classes: Classes which can be predicted
    :param cap: Capture Device
    :param tracker: Hand Tracker
    :param model_path: Path to the model
    :return:
    """
    # model to predict on
    model: Sequential = load_model(model_path)
    # NB! number of outputs of the model MUST be the same length as classes

    while True:
        # read frames from device capture
        _, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 50  # Why the fuck did I do -50, I don't fucking know

        tracker.hands_finder(frame)
        lms = tracker.position_finder(frame, normalized=True)

        if lms:
            landmark = np.array(lms).reshape((42,))

            # get probability of the classes predicted
            predictions = model.predict(np.array([landmark]))
            # get the highest scored predicted index
            predicted = np.argmax(np.squeeze(predictions))
            if np.squeeze(predictions)[predicted] > 0:
                predicted_class = classes[predicted]
                cv2.putText(frame, predicted_class, (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5, thickness=2, color=(0, 0, 0))
                for i, prediction in enumerate(np.squeeze(predictions)):
                    cv2.putText(frame, '{0:.2f}%'.format(prediction * 100), org=(0, 20 + int(height - (i * 10))),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=0.25)
                    cv2.rectangle(frame, (50, 15 + int(height - (i * 10))),
                                  (50 + int((width - 50) * prediction), 20 + int(height - (i * 10))),
                                  color=(int(255 * prediction), int(255 * prediction), int(255 * prediction)),
                                  thickness=-1)
                    cv2.putText(frame, f'{classes[i]}',
                                org=(50 + int((width - 50) * prediction), 20 + int(height - (i * 10))),
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
    """
    Projects a hand onto a canvas / window with constant height and width.


    :param cap:
    :param tracker:
    :return:
    """

    # Define webcam (capture device) height and width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a sharp filter (kernel) to sharpen image after being projected
    sharpen = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    while True:
        # read frame from capture device
        _, frame = cap.read()

        # define bounding box dimensions
        high: list = [0, 0]
        low: list = [width, height]

        # find hands inside a frame
        tracker.hands_finder(frame, draw=False)

        # find all the landmarks for the hand
        landmarks = tracker.position_finder(frame, hand_no=0)

        # find the lowest and highest points for the bounding box
        for lm in landmarks:
            if lm[0] < low[0]:
                low[0] = lm[0]
            if lm[1] < low[1]:
                low[1] = lm[1]
            if lm[0] > high[0]:
                high[0] = lm[0]
            if lm[1] > high[1]:
                high[1] = lm[1]

        # draw the bounding box
        cv2.rectangle(frame, low, high, color=(0, 0, 255))

        pts1 = np.float32([low, [high[0], low[0]], [low[0], high[1]], high])
        pts2 = np.float32([[0, 0], [high[0] - low[0], 0], [0, high[1] - low[1]], [high[0] - low[0], high[1] - low[1]]])
        pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        result = cv2.warpPerspective(frame, matrix, (400, 600))
        cv2.imshow('frame', frame)
        # cv2.imshow('projection', result)
        if ((high[0] - low[0]) > 0) and ((high[1] - low[1]) > 0):
            # crop image based on highest and lowest value
            cropped = frame[low[1]:high[1], low[0]:high[0]]
            # resize image such that the dimensions of x & y will be the same
            cropped = cv2.resize(cropped, (400, 400))
            # apply the sharp filter to the resized image
            cropped = cv2.filter2D(cropped, -1, sharpen)
            # cropped = tracker.hands_finder(cropped, draw=True)
            cv2.imshow('cropped', cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def triangle_detection(cap: cv2.VideoCapture):
    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            print(len(approx))
            if len(approx) == 3:
                frame = cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)
        cv2.imshow('Detected', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    # train(cap, tracker)
    # predict_gesture(cap, tracker, './models/4_model.h5', ['down', 'stop', 'left', 'right', 'up', 'down', 'pinch'])
    # projection(cap, tracker)
    triangle_detection(cap)


def landmark_to_distance(landmarks: list):
    """
    Returns a list of distance between point 0 to all other points
    :param landmarks:
    :return:
    """

    result = list()

    result.extend([float(landmarks[0]) - float(landmarks[2]), float(landmarks[1]) - float(landmarks[3])])
    for i in range(4, len(landmarks), 2):
        result.append(
            math.dist([float(landmarks[0]), float(landmarks[1])], [float(landmarks[i]), float(landmarks[i + 1])]))
    return result


if __name__ == '__main__':
    main()

# Labels
# 0 = stop, 1 = pinch, 2 = up, 3 = down, 4 = left, 5 = right, 6 = peace
