from tensorflow.keras.models import load_model, Sequential
from typing import List, Union, Tuple
from enum import Enum

import face_recognition
import mediapipe as mp
import numpy as np
import keras
import cv2


class ControlType(Enum):
    """
    Enumerator class to switch between controller modes.
    """
    controller = 0  # xbox_controller
    keyboard = 1
    gesture = 2


class HandTracker:
    """
    The HandTracker object can detect hands in an image and return
    the joint position of the hand.

    :param max_hands: max hand returned in an image
    :type max_hands: int
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, model_complexity=1, track_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.detectionCon = detection_con
        self.modelComplex = model_complexity
        self.trackCon = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def hands_finder(self, image, draw=True):
        """
        Find number of hands in an image and save the result.

        :param image: Image to operate on
        :param draw: Choose if joints should be drawn on image
        :return: return image as numpy.ndarray
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def position_finder(self, image, hand_no=0, draw: bool = False, append_id: bool = False, normalized: bool = False):
        """
        Returns hand joints in image

        :param image: image to operate on
        :param hand_no: index of hand to operate on
        :param draw: Should the function draw the joints on the image
        :param append_id: if the hand joint id should be appended to the list
        :return: List
        """
        lmlist = []
        cx, cy = 0, 0
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape

                if normalized:
                    cx, cy = lm.x, lm.y
                else:
                    cx, cy = int(lm.x * w), int(lm.y * h)

                if append_id:
                    lmlist.append([id, cx, cy])
                else:
                    lmlist.append([cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return lmlist


class GestureControl:
    """
    The GestureControl object controls gesture recognition.
    
    :param model: The model is used to locate where the model is.
    :type model: str
    :type model: str
    :param classes: Classes is used for the classification
    :
    """

    def __init__(self, model: str, classes: List[str], hand_tracker: HandTracker = HandTracker()):
        self.model: Sequential = load_model(model)
        self.classes = classes
        self.hand_tracker = hand_tracker

    def predict(self, frame: np.ndarray, debug: bool = False, return_predictions: bool = False):
        """
        Predict if what class is in a frame

        :param return_predictions: Whetever to return index or class prediction
        :param debug: Print time used to predict a frame
        :type debug: bool
        :param frame: Ndarray to operate on
        :return: class predicted
        :rtype: string
        """
        self.hand_tracker.hands_finder(frame, draw=False)
        landmarks = self.hand_tracker.position_finder(frame, normalized=True)
        if not landmarks:
            return None, None
        landmark = np.array(landmarks).reshape((42,))
        predictions = self.model.predict(np.array([landmark]))

        class_id = np.argmax(np.squeeze(predictions))
        if return_predictions:
            return predictions
        return self.classes[class_id], np.max(np.squeeze(predictions))


def detect_qr_code(frame: np.ndarray):
    """
    Detect a QR code in an image and draw bounding boxes

    :param frame: Image to operate on
    :return: Image as numpy.ndarray with bounding boxes
    """

    image: np.ndarray = frame.copy()
    detector = cv2.QRCodeDetector()

    # image = cv2.cvtColor(src=image, dst=image, code=cv2.COLOR_BGR2GRAY)
    text, points, _ = detector.detectAndDecode(image)

    if points is not None:
        for point in points[0]:
            image = cv2.circle(image, (int(point[0]), int(point[1])), 3, thickness=3, color=(0, 255, 0))
        image = cv2.polylines(image, np.int32([np.array(points[0])]), color=(0, 255, 0), isClosed=True, thickness=2)

    return image


def detect_face(frame: np.ndarray, draw: bool = False) -> Union[Union[Tuple, Tuple, Tuple], Union[None, None, None]]:
    """
    Function to detect a face in an image
    :param draw: If the function should draw the bounding box of the detected face
    :param frame: Frame to find face
    :return: (x1, y1) & (x2, y2) to represent a bounding box
    """
    face = face_recognition.face_locations(frame)
    if len(face) > 0:
        face = face[0]
        if draw:
            cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)
        return (face[3], face[0]), (face[1], face[2]), \
               (face[1] - int((face[1] - face[3]) / 2), face[2] - int((face[2] - face[0]) / 2))
    return None, None, None


def in_boundary(delta_x: int, delta_y: int, frame_center: tuple, point: tuple) -> Union[bool, Tuple]:
    if None in [delta_x, delta_y, frame_center, point]:
        return False, None

    if frame_center[0] - delta_x < point[0] < frame_center[0] + delta_x:
        if frame_center[1] - delta_y < point[1] < frame_center[1] + delta_y:
            return True, None
    return False, (frame_center[0] - delta_x > point[0], frame_center[1] - delta_y > point[1],
                   frame_center[0] + delta_x < point[0], frame_center[1] + delta_y < point[1])


def prediction_statistics(frame: np.ndarray, tracker: GestureControl, model: Sequential, classes: list):
    width = frame.shape[1]
    height = frame.shape[0] - 50

    lms = tracker.hand_tracker.position_finder(frame, normalized=True)

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
                              (50 + int((width - 100) * prediction), 20 + int(height - (i * 10))),
                              color=(int(255 * prediction), int(255 * prediction), int(255 * prediction)),
                              thickness=-1)
                cv2.putText(frame, f'{classes[i]}',
                            org=(60 + int((width - 100) * prediction), 20 + int(height - (i * 10))),
                            fontScale=0.25, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))