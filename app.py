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
from utils import HandTracker
from threading import Thread
from tellogui import *

"""
NOT AN IMPORTANT FILE, LOOK AWAY FROM THIS

Integrated OpenCv with tkinter
"""


class MainWindow(QMainWindow):
    """
    Tello video stream and control integrated into a tkinter window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tello = Tello()
        # self.tello.connect()
        self.setGeometry(640, 480, 640, 480)
        self.container = QVBoxLayout(self)

        # tello preview
        self.tello_preview = QLabel(self)
        # webcam preview
        self.webcam_preview = QLabel(self)

        # tello feed worker
        self.tello_feed = VideoFeed(self)
        # webcam feed worker
        self.camera_feed = CameraFeed(self)

        self.feed_widget = QWidget()
        self.feed_layout = QHBoxLayout(self.feed_widget)

        self.feed_layout.addWidget(self.tello_preview, alignment=Qt.AlignTop)
        self.feed_layout.addWidget(self.webcam_preview, alignment=Qt.AlignTop)

        self.panel = QWidget()
        self.panel_layout = QGridLayout(self.panel)

        self.controller = QComboBox()
        self.panel_layout.addWidget(self.controller, 0, 0)

        self.container.addWidget(self.feed_widget, alignment=Qt.AlignLeft)
        self.container.addWidget(self.panel, alignment=Qt.AlignTop)

        widget = QWidget()
        widget.setLayout(self.container)

        self.setCentralWidget(widget)
        self.start()

    def start(self):
        self.tello_feed.start()
        self.tello_feed.image_update.connect(self.update_tello_preview)

        self.camera_feed.start()
        self.camera_feed.image_update.connect(self.update_webcam_preview)

    def update_tello_preview(self, image):
        self.tello_preview.setPixmap(QPixmap.fromImage(image))

    def update_webcam_preview(self, image):
        self.webcam_preview.setPixmap(QPixmap.fromImage(image))


class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self, parent: MainWindow):
        super(CameraFeed, self).__init__(parent)
        self.active = True
        self.hand_tracker = HandTracker()

    def run(self):
        self.active = True

        cap = cv2.VideoCapture(0)

        while self.active:
            _, frame = cap.read()

            frame = self.hand_tracker.hands_finder(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            qt_img = qt_img.scaled(320, 240, Qt.KeepAspectRatioByExpanding)
            self.image_update.emit(qt_img)

    def stop(self):
        self.active = False


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

                frame: np.ndarray = read.frame
                # frame = self.hand_tracker.hands_finder(frame)

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
                draw_rotation(frame, self.tello.get_yaw())
                draw_roll(frame, self.tello.get_roll())
                draw_pitch(frame, self.tello.get_pitch())
                draw_velocity(frame, vel)

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
                qt_img = qt_img.scaled(320, 240, Qt.KeepAspectRatioByExpanding)
                self.image_update.emit(qt_img)
            except Exception as error:
                print(error)

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
