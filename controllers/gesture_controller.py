from djitellopy import Tello
from utils import GestureControl
import numpy as np


def gesture_controller(frame: np.ndarray, tello: Tello, gesture_control: GestureControl, threshold: float = 0.5,
                       debug: bool = False):
    """
    Control the drone using gestures.

    :param threshold: Minimum value / probability for drone to react to gesture
    :param debug: Decide if the predicted gesture should be printed
    :param frame: Image to predict the frame on
    :type frame: np.ndarray
    :param tello: The Tello drone / object to perform the instructions on
    :type tello: Tello
    :param gesture_control: GestureControl object which performs the guessing
    """

    # key value pair for gesture -> command
    command = {
        'up': f'rc {0} {0} {50} {0}', 'down': f'rc {0} {0} {-50} {0}',
        'left': f'rc {-50} {0} {0} {0}', 'right': f'rc {50} {0} {0} {0}'
    }

    gesture, probability = gesture_control.predict(frame)
    print(probability)
    if gesture and probability >= threshold:
        try:
            if debug:
                print(f'[DEBUG] Gesture ➡ {gesture}', end='\r')

            if gesture == 'stop':  # takeoff / land
                if tello.is_flying:
                    tello.land()
                else:
                    tello.takeoff()
            elif gesture in command.keys():
                tello.send_command_without_return(command[gesture])
            else:
                tello.send_command_without_return('rc 0 0 0 0')
        except Exception as error:
            print(f'[ERROR]: {error}')
    elif tello.is_flying:
        tello.send_command_without_return('rc 0 0 0 0')