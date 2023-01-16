import time
import random

import cv2
import mediapipe as mp
from djitellopy import Tello
from threading import Thread
from queue import Queue
from controller import XboxController
import numpy as np
from utils import HandTracker


def video_stream(tello: Tello, queue: Queue):
    tello.streamon()
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_bitrate(tello.BITRATE_5MBPS)
    tello.set_video_resolution(tello.RESOLUTION_480P)
    tracker = HandTracker()
    read = tello.get_frame_read()
    base_vector = [4, 2, 4]
    prev_frame_time = 0
    while True:
        # velocity
        try:
            vel = [tello.get_speed_x(), tello.get_speed_y(), tello.get_speed_z()]

            img = read.frame
            img = tracker.hands_finder(img)

            center = [int(img.shape[1] / 2), int(img.shape[0] / 2)]

            # crosshair
            cv2.circle(img=img, center=(int(img.shape[1] / 2), int(img.shape[0] / 2)), color=(255, 255, 255), radius=2)

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


def draw_rotation(img: np.ndarray, degree):
    center = (img.shape[1] - 50, img.shape[0] - 50)
    cv2.circle(img, center=center, radius=30, color=(255, 0, 255))
    cv2.circle(img, center=(center[0] + int(np.cos(np.deg2rad(degree)) * 30),
                            int(center[1] + np.sin(np.deg2rad(degree)) * 30)),
               thickness=4, radius=4, color=(255, 255, 255))


def draw_roll(img: np.ndarray, degree):
    left = (img.shape[1] - 80, img.shape[0] - 150)
    cv2.line(img, pt1=left, pt2=[img.shape[1] - 20, img.shape[0] - 150], color=(255, 255, 255))
    cv2.circle(img, center=(img.shape[1] - 50 + int(-(60 * degree / 360)), img.shape[0] - 150), color=(0, 0, 255),
               radius=3, thickness=3)


def draw_pitch(img: np.ndarray, degree):
    bottom = (img.shape[1] - 50, img.shape[0] - 200)
    cv2.line(img, pt1=bottom, pt2=[bottom[0], img.shape[0] - 260], color=(255, 255, 255))
    cv2.circle(img, center=(bottom[0], img.shape[0] - 230 + int(-(60 * degree / 180))), color=(0, 0, 255),
               radius=3, thickness=3)


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


def clear_queue(queue: Queue):
    while not queue.empty():
        queue.get()


def keep_alive(tello: Tello):
    while True:
        tello.send_command_with_return('stop')
        time.sleep(10)


def controller(tello: Tello, queue: Queue):
    downwards_cam = False
    using_controller = True
    if using_controller:
        joy = XboxController()
    tello.set_speed(100)
    while True:
        if using_controller:
            input = joy.read()
            try:
                tello.send_command_without_return(
                    f"rc {int(100 * input['LJ_X'])} {int(100 * input['LJ_Y'])} {int(100 * input['RJ_Y'])} {int(100 * input['RJ_X'])}")
                time.sleep(0.01)

                if input['RJ_T']:
                    if tello.is_flying:
                        tello.land()
                    else:
                        tello.takeoff()
                elif input['X']:
                    tello.send_command_without_return('flip l')
                elif input['B']:
                    tello.send_command_without_return('flip r')
                elif input['A']:
                    tello.send_command_without_return('flip b')
                elif input['Y']:
                    tello.send_command_without_return('flip f')
                elif input['RB']:
                    tello.set_video_direction(int(downwards_cam))
                    downwards_cam = not downwards_cam
            except Exception as exception:
                print(exception)
        else:
            key = queue.get()
            try:
                if key == ord('w'):
                    tello.send_command_without_return('rc 0 100 0 0')
                elif key == ord('s'):
                    # tello.send_command_with_return(f'back {sensitivity}', timeout)
                    tello.send_command_without_return('rc 0 -100 0 0')
                elif key == ord('a'):
                    tello.send_command_without_return('rc -100 0 0 0')
                    # tello.send_command_with_return(f'left {sensitivity}', timeout)
                elif key == ord('d'):
                    tello.send_command_without_return('rc 100 0 0 0')
                    # tello.send_command_with_return(f'right {sensitivity}', timeout)
                elif key == ord('q'):
                    break
                elif key == ord('r'):
                    tello.send_command_without_return('rc 0 0 0 0')
                elif key == ord('x'):
                    tello.send_command_without_return('rc 0 0 0 0')
                    if tello.is_flying:
                        tello.land()
                    else:
                        tello.send_command_without_return('takeoff')
                clear_queue(queue)
            except Exception as e:
                print(f'[INFO] Command Failed: {e}')
    tello.land()
    cv2.destroyWindow('DRONE - FEED')


def main():
    queue = Queue()
    tello = Tello()
    tello.connect()

    threads = [
        Thread(target=controller, args=(tello, queue)),
        Thread(target=video_stream, args=(tello, queue))]

    for thread in threads:
        thread.start()


if __name__ == '__main__':
    main()
