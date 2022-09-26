import time

from PyQt5 import QtCore, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from djitellopy import Tello
import mediapipe as mp
import numpy as np
import sys
import cv2


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


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tello = Tello()
        self.tello.connect()
        self.setGeometry(640, 480, 640, 480)
        self.container = QHBoxLayout(self)

        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignTop)
        # feed worker
        self.feed = VideoFeed(self)
        self.start()

        self.setCentralWidget(self.preview)
        self.show()

    def start(self):
        self.feed.start()
        self.feed.image_update.connect(self.update_image)

    def update_image(self, image):
        self.preview.setPixmap(QPixmap.fromImage(image))


class VideoFeed(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        try:
            self.hand_tracker = HandTracker()
            self.base_vector = [4, 2, 4]
            self.tello = parent.tello
            self.active = False
        except Exception as error:
            print(error)

    def run(self):
        self.active = True
        self.tello.streamon()
        self.tello.set_video_fps(self.tello.FPS_30)
        self.tello.set_video_bitrate(self.tello.BITRATE_5MBPS)
        self.tello.set_video_resolution(self.tello.RESOLUTION_480P)

        prev_frame_time = 0
        read = self.tello.get_frame_read()
        while self.active:
            try:
                vel = [self.tello.get_speed_x(), self.tello.get_speed_y(), self.tello.get_speed_z()]

                frame = read.frame
                frame = self.hand_tracker.hands_finder(frame)

                center = [int(frame.shape[1] / 2), int(frame.shape[0] / 2)]
                # GRAPHIC START

                cv2.circle(img=frame, center=(int(frame.shape[1] / 2), int(frame.shape[0] / 2)), color=(255, 255, 255),
                           radius=2)

                cv2.putText(frame, f"Battery: {str(self.tello.get_battery())}%", (20, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                            fontScale=0.5,
                            thickness=1)
                cv2.putText(frame, f"Airborne: {self.tello.is_flying}", (20, 80),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                            fontScale=0.5, thickness=1)
                cv2.putText(frame, f"Velocity:({str(vel)})",
                            (20, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                            fontScale=0.5, thickness=1)

                vel = np.multiply(vel, self.base_vector)
                cv2.arrowedLine(img=frame, pt1=[50, 200], pt2=[50 + vel[0], 200 - vel[2]],
                                thickness=2, color=(0, 255, 0))

                # DRAW ROTATIONS
                self.draw_rotation(frame, self.tello.get_yaw())
                self.draw_roll(frame, self.tello.get_roll())
                self.draw_pitch(frame, self.tello.get_pitch())
                self.draw_velocity(frame, vel)

                # CALCULATE FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                # CALCULATE FPS
                cv2.putText(img=frame, text=f'FPS: {fps}', org=(frame.shape[1] - 150, 40), color=(255, 255, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)
                cv2.putText(img=frame, text=f'Temp: {str(self.tello.get_temperature())}C',
                            org=(frame.shape[1] - 150, 80), color=(255, 255, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)


                # GRAPHIC END
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                qt_img = qt_img.scaled(640, 480, Qt.KeepAspectRatio)
                self.image_update.emit(qt_img)
            except Exception as error:
                print(error)

    @staticmethod
    def draw_rotation(img: np.ndarray, degree):
        center = (img.shape[1] - 50, img.shape[0] - 50)
        cv2.circle(img, center=center, radius=30, color=(255, 255, 255))
        cv2.circle(img, center=(center[0] + int(np.cos(np.deg2rad(degree)) * 30),
                                int(center[1] + np.sin(np.deg2rad(degree)) * 30)),
                   thickness=4, radius=4, color=(0, 0, 255))

    @staticmethod
    def draw_roll(img: np.ndarray, degree):
        left = (img.shape[1] - 80, img.shape[0] - 150)
        cv2.line(img, pt1=left, pt2=[img.shape[1] - 20, img.shape[0] - 150], color=(255, 255, 255))
        cv2.circle(img, center=(img.shape[1] - 50 + int(-(60 * degree / 360)), img.shape[0] - 150), color=(0, 0, 255),
                   radius=3, thickness=3)

    @staticmethod
    def draw_pitch(img: np.ndarray, degree):
        bottom = (img.shape[1] - 50, img.shape[0] - 200)
        cv2.line(img, pt1=bottom, pt2=[bottom[0], img.shape[0] - 260], color=(255, 255, 255))
        cv2.circle(img, center=(bottom[0], img.shape[0] - 230 + int(-(60 * degree / 180))), color=(0, 0, 255),
                   radius=3, thickness=3)

    @staticmethod
    def draw_velocity(img: np.ndarray, vel):
        cv2.arrowedLine(img=img, pt1=[30, img.shape[0] - 30],
                        pt2=[30 + vel[0], img.shape[0] - 30],
                        color=(0, 0, 255), thickness=2)
        cv2.arrowedLine(img=img, pt1=[30, img.shape[0] - 30],
                        pt2=[30, img.shape[0] - 30 - vel[2]],
                        color=(255, 0, 0), thickness=2)
        cv2.arrowedLine(img=img, pt1=[30, img.shape[0] - 30],
                        pt2=[30 + vel[1], img.shape[0] - 30 - vel[1]],
                        color=(255, 255, 255), thickness=1)
        cv2.circle(img, (30, img.shape[0] - 30), color=(0, 255, 0), thickness=1, radius=2)

    def stop(self):
        self.tello.streamoff()
        self.active = False


def main():
    app = QApplication(sys.argv)

    _app = MainWindow()
    _app.show()

    app.exec()


if __name__ == '__main__':
    main()
