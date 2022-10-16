from djitellopy import Tello
from utils import GestureControl
import numpy as np


def gesture_controller(frame: np.ndarray, tello: Tello, gesture_control: GestureControl, debug: bool = False):
    """
    Control the drone using gestures.

    :param debug: Decide if the predicted gesture should be printed
    :param frame: Image to predict the frame on
    :type frame: np.ndarray
    :param tello: The Tello drone / object to perform the instructions on
    :type tello: Tello
    :param gesture_control: GestureControl object which performs the guessing
    """

    command = {
        'thumbs up': f'rc {0} {0} {10} {0}', 'thumbs down': f'rc {0} {0} {-10} {0}',
        'peace': f'rc {0}  {-10} {0} {0}', 'okay': f'rc {0} {10} {0} {0}'
    }

    gesture = gesture_control.predict(frame)

    if gesture:
        try:
            if debug:
                print(f'[DEBUG] Gesture ➡ {gesture}', end='\r')

            if gesture == 'rock':  # takeoff / land
                if tello.is_flying:
                    tello.send_command_without_return('land')
                else:
                    tello.send_command_without_return('takeoff')
            elif gesture in command.keys():
                tello.send_command_without_return(command[gesture])
        except Exception as error:
            print(f'[ERROR]: {error}')
