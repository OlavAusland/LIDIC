import mediapipe as mp
import cv2
import numpy as np


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

    def position_finder(self, image, hand_no=0, draw=True):
        lmlist = []
        cx, cy = 0, 0
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return lmlist


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