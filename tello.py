import time

import cv2
from djitellopy import Tello
from threading import Thread
from queue import Queue
from controllers.controller import XboxController
import numpy as np
from utils import GestureControl, HandTracker, ControlType
from tellogui import *
from controllers.gesture_controller import gesture_controller
from controllers.keyboard_controller import keyboard_controller


EXIT = False


def video_stream(tello: Tello, queue: Queue, frame_queue: Queue):
    global EXIT
    tello.streamon()
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_bitrate(tello.BITRATE_AUTO)
    tello.set_video_resolution(tello.RESOLUTION_720P)  # 1280 x 720 pixels

    tracker = HandTracker()
    read = tello.get_frame_read()
    base_vector = [4, 2, 4]
    prev_frame_time = 0
    while True:
        if EXIT:
            break
        # velocity
        try:
            vel = [tello.get_speed_x(), tello.get_speed_y(), tello.get_speed_z()]

            img = read.frame  # requests frame from drone real time
            frame_queue.put(img)

            img = tracker.hands_finder(img)

            center = (int(img.shape[1] / 2), int(img.shape[0] / 2))

            # crosshair
            cv2.circle(img=img, center=center, color=(255, 255, 255), radius=2)

            cv2.putText(img, f"Battery: {str(tello.get_battery())}%", (20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5,
                        thickness=1)
            cv2.putText(img, f"Airborne: {tello.is_flying}", (20, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5, thickness=1)
            cv2.putText(img, f"Velocity:({str(vel)})",
                        (20, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                        fontScale=0.5, thickness=1)

            vel = np.multiply(vel, base_vector)
            cv2.arrowedLine(img=img, pt1=[50, 200], pt2=[50 + vel[0], 200 - vel[2]],
                            thickness=2, color=(0, 255, 0))

            # DRAW ROTATIONS
            draw_rotation(img, tello.get_yaw())
            draw_roll(img,  tello.get_roll())
            draw_pitch(img, tello.get_pitch())
            draw_velocity(img, vel)

            # CALCULATE FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            # CALCULATE FPS
            cv2.putText(img=img, text=f'FPS: {fps}', org=(img.shape[1] - 150, 40), color=(255, 255, 255),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)
            cv2.putText(img=img, text=f'Temp: {str(tello.get_temperature())}C',
                        org=(img.shape[1] - 150, 80), color=(255, 255, 255),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)

            cv2.imshow('DRONE - FEED', img)
            # control
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                queue.put(key)
        except Exception as error:
            print(error)


def clear_queue(queue: Queue):
    while not queue.empty():
        queue.get()


def keep_alive(tello: Tello):
    while True:
        tello.send_command_with_return('stop')
        time.sleep(10)


def controller(tello: Tello, key_queue: Queue, frame_queue: Queue):
    global EXIT
    joy = None
    downwards_cam = False
    tello.set_speed(100)

    control_type = ControlType.gesture  # change control type of tello

    # classes = ['down', 'stop', 'left', 'right', 'up', 'down', 'pinch']
    classes = ['stop', 'undefined', 'up', 'down', 'left', 'right', 'undefined']
    gesture_control = GestureControl('models/7_model.h5', classes)

    if control_type == control_type.controller:
        joy = XboxController()

    cap = cv2.VideoCapture(0)
    while True:
        if control_type == ControlType.controller and (joy is not None):
            _input = joy.read()
            try:
                tello.send_command_without_return(
                    f"rc {int(100 * _input['LJ_X'])} {int(100 * _input['LJ_Y'])} {int(100 * _input['RJ_Y'])} {int(100 * _input['RJ_X'])}")
                time.sleep(0.01)

                if _input['RJ_T']:
                    if tello.is_flying:
                        tello.land()
                    else:
                        tello.takeoff()
                elif _input['X']:
                    tello.send_command_without_return('flip l')
                elif _input['B']:
                    tello.send_command_without_return('flip r')
                elif _input['A']:
                    tello.send_command_without_return('flip b')
                elif _input['Y']:
                    tello.send_command_without_return('flip f')
                elif _input['RB']:
                    tello.set_video_direction(int(downwards_cam))
                    downwards_cam = not downwards_cam
            except Exception as exception:
                print(exception)
        elif control_type == ControlType.gesture:
            _, frame = cap.read()  # -> Use webcam frame
            drone_frame = frame_queue.get()  # -> Use drone frame
            gesture_controller(frame=frame, tello=tello,
                               gesture_control=gesture_control, debug=True)
            cv2.imshow('webcam', frame)
        elif control_type == ControlType.keyboard:
            keyboard_controller(tello=tello, key_queue=key_queue)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            EXIT = True
            break
    tello.land()
    cv2.destroyWindow('DRONE - FEED')


def main():
    frame = Queue()
    key_queue = Queue()

    tello = Tello()
    tello.connect()

    threads = [
        Thread(target=controller, args=(tello, key_queue, frame)),
        Thread(target=video_stream, args=(tello, key_queue, frame))]

    for thread in threads:
        thread.start()


if __name__ == '__main__':
    main()
