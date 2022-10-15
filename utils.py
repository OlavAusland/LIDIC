from typing import List

import keras
import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class HandTracker:
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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def position_finder(self, image, hand_no=0, draw=True, append_id: bool = False):
        lmlist = []
        cx, cy = 0, 0
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape
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
    :type string:
    :type model: str
    :param classes: Classes is used for the classification
    :
    """
    def __init__(self, model: str, classes: List[str], hand_tracker: HandTracker = HandTracker()):
        self.model: keras.Sequential = load_model(model)
        self.classes = classes
        self.hand_tracker = hand_tracker

    def predict(self, frame: np.ndarray):
        self.hand_tracker.hands_finder(frame, draw=False)
        landmarks = self.hand_tracker.position_finder(frame, hand_no=0, draw=False)
        print('Here')
        if not landmarks:
            return 'None'

        prediction = self.model.predict([landmarks])

        class_id = np.argmax(prediction)
        return self.classes[class_id]


def detectQRCode(frame: np.ndarray):
    image: np.ndarray = frame.copy()
    detector = cv2.QRCodeDetector()

    image = cv2.cvtColor(src=image, dst=image, code=cv2.COLOR_BGR2GRAY)
    text, points, _ = detector.detectAndDecode(image)

    if points is not None:
        for point in points[0]:
            image = cv2.circle(image, (int(point[0]), int(point[1])), 3, thickness=3, color=(0, 255, 0))
        image = cv2.polylines(image, np.int32([np.array(points[0])]), color=(0, 255, 0), isClosed=True, thickness=2)
    return image