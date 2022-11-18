import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from controllers import keyboard_controller, gesture_controller, controller
from utils import HandTracker, ControlType
from djitellopy import Tello
from threading import Thread
from PIL import ImageTk, Image
import tkinter as tk
from tellogui import *
import time
import cv2


class Application(tk.Frame):
    def __init__(self, master=None):
        super(Application, self).__init__(master)
        self.grid()

        self.controller_type: ControlType = ControlType.gesture
        self.tello = Tello()
        self.tello.connect()

        self.webcam_stream = VideoCapture(0)
        self.tello_stream = TelloCapture(tello=self.tello)

        self.webcam_feed = tk.Label(self)
        self.webcam_feed.grid(row=0, column=0, columnspan=12, rowspan=6, sticky='NSEW')

        self.tello_feed = tk.Label(self)
        self.tello_feed.grid(row=6, column=0, columnspan=12, rowspan=6, sticky='NSEW')

        Thread(target=self.update_webcam, args=(0.01,)).start()
        Thread(target=self.update_tello_stream, args=(0.01,)).start()

        self.connect = tk.Button(self, text='Connect')
        self.connect.grid(row=13, column=0, rowspan=1, columnspan=12, sticky='NSEW')

    def update_webcam(self, delay: int):
        while True:
            ret, frame = self.webcam_stream.get_feed()

            if ret:
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.webcam_feed.imgtk = imgtk
                self.webcam_feed.configure(image=imgtk)
            time.sleep(delay)

    def update_tello_stream(self, delay: int):
        while True:
            ret, frame = self.tello_stream.get_frame()

            if ret:
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.tello_feed.imgtk = imgtk
                self.tello_feed.configure(image=imgtk)
            time.sleep(delay)


class Statistics(tk.Frame):
    def __init__(self, master=None):
        super(Statistics, self).__init__(master)

        self.figure = Figure(figsize=(5, 4), dpi=90)
        self.plot = self.figure.add_subplot()
        self.labels = ['left', 'right', 'up', 'down', 'stop']
        self.data = np.random.uniform(low=0.5, high=13, size=(5,))

        self.plot.bar(self.labels, self.data, 0.25)

        self.canvas = FigureCanvasTkAgg(self.figure)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=13, columnspan=12, rowspan=6, sticky='WEN')


class TelloCapture:
    def __init__(self, tello: Tello):
        self.base_vector = [4, 2, 4]
        self.tracker = HandTracker()
        self.tello = tello
        self.tello.set_video_resolution(tello.RESOLUTION_480P)
        self.tello.streamon()
        self.read = tello.get_frame_read()

    def get_frame(self):
        try:
            velocity = [self.tello.get_speed_x(), self.tello.get_speed_y(), self.tello.get_speed_z()]
            frame = self.read.frame

            frame = self.tracker.hands_finder(frame)

            center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))

            cv2.circle(img=frame, center=center, color=(255, 255, 255), radius=2)

            cv2.putText(frame, f"Battery: {str(self.tello.get_battery())}%", (20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5,
                        thickness=1)
            cv2.putText(frame, f"Airborne: {self.tello.is_flying}", (20, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5, thickness=1)
            cv2.putText(frame, f"Velocity:({str(velocity)})",
                        (20, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5, thickness=1)

            velocity = np.multiply(velocity, self.base_vector)
            cv2.arrowedLine(img=frame, pt1=[50, 200], pt2=[50 + velocity[0], 200 - velocity[2]],
                            thickness=2, color=(0, 255, 0))

            # DRAW ROTATION
            draw_rotation(frame, self.tello.get_yaw())
            draw_roll(frame, self.tello.get_roll())
            draw_pitch(frame, self.tello.get_pitch())
            draw_velocity(frame, velocity)
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except ValueError as error:
            print(f'{error}')
            return None, None


class VideoCapture:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError('Unable to open capture device!', source)

        self.cap_info = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(self.cap.get(cv2.CAP_PROP_FPS)))

    def get_feed(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return None, None

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def main():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()


if __name__ == '__main__':
    main()
