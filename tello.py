import time
import cv2
import mediapipe as mp
from djitellopy import Tello
from threading import Thread
from queue import Queue
from controller import XboxController
import numba


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


def video_stream(tello: Tello, queue: Queue):
    tello.streamon()
    # tello.set_video_bitrate(Tello.BITRATE_1MBPS)
    # tello.set_video_fps(Tello.FPS_15)
    tracker = HandTracker()
    read = tello.get_frame_read()

    while True:
        img = read.frame
        img = tracker.hands_finder(img)
        cv2.putText(img, str(tello.get_battery()), (40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                    fontScale=1,
                    thickness=2)
        cv2.imshow('DRONE - FEED', img)

        # control
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            queue.put(key)


def controller(tello: Tello, queue: Queue):
    using_controller = True

    joy = XboxController()
    while True:

        if using_controller:
            input = joy.read()
            try:
                tello.send_command_without_return(f"rc {int(100 * input['LJ_X'])} {int(100 * input['LJ_Y'])} {int(100 * input['RJ_Y'])} {int(100 * input['RJ_X'])}")
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
                        tello.takeoff()
                clear_queue(queue)
            except Exception as e:
                print(f'[INFO] Command Failed: {e}')
    tello.land()
    cv2.destroyWindow('DRONE - FEED')


def clear_queue(queue: Queue):
    while not queue.empty():
        queue.get()


def keep_alive(tello: Tello):
    while True:
        tello.send_command_with_return('stop')
        time.sleep(10)


def main():
    queue = Queue()
    tello = Tello()
    tello.connect()

    threads = [
        Thread(target=video_stream, args=(tello, queue)),
        Thread(target=controller, args=(tello, queue))]

    for thread in threads:
        thread.start()


if __name__ == '__main__':
    main()
