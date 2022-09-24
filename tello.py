import time
import cv2
import keyboard
import mediapipe as mp
from queue import Queue
from djitellopy import Tello
from threading import Thread


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


def video_stream(tello: Tello):
    tello.streamon()
    # tello.set_video_bitrate(Tello.BITRATE_1MBPS)
    # tello.set_video_fps(Tello.FPS_15)
    frame_read = tello.get_frame_read()
    tracker = HandTracker()

    while True:
        img = frame_read.frame
        img = tracker.hands_finder(img)
        cv2.putText(img, str(tello.get_battery()), (40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                    fontScale=1,
                    thickness=2)
        cv2.imshow('DRONE - FEED', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def controller(tello: Tello):
    tello.takeoff()
    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            tello.move_forward(20)
        elif key == ord('s'):
            tello.move_back(20)
        elif key == ord('a'):
            tello.move_left(20)
        elif key == ord('d'):
            tello.move_right(20)
        elif key == ord('q'):
            tello.move_up(20)
        elif key == ord('r'):
            tello.move_down(20)

    tello.land()
    cv2.destroyWindow('DRONE - FEED')


def keep_alive(tello: Tello):
    while True:
        tello.send_command_with_return('stop')
        time.sleep(10)


def main():
    tello = Tello()
    tello.connect()

    threads = [Thread(target=keep_alive, args=(tello, )),
               Thread(target=video_stream, args=(tello, )),
               Thread(target=controller, args=(tello, ))]

    for thread in threads:
        thread.start()


if __name__ == '__main__':
    main()
