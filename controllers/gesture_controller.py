from djitellopy import Tello
from utils import GestureControl
import numpy as np
from tensorflow.keras.models import load_model, Sequential


def gesture_controller(frame: np.ndarray, tello: Tello, gesture_control: GestureControl, model: Sequential,
                       threshold: float = 0.5,
                       debug: bool = False):
    """
    Control the drone using gestures.

    :param model: Model used for prediction
    :param threshold: Minimum value / probability for drone to react to gesture
    :param debug: Decide if the predicted gesture should be printed
    :param frame: Image to predict the frame on
    :type frame: np.ndarray
    :param tello: The Tello drone / object to perform the instructions on
    :type tello: Tello
    :param gesture_control: GestureControl object which performs the guessing
    """

    tracker = gesture_control.hand_tracker

    commands = {
        '4': f'rc {-25} {0} {0} {0}', '2': f'rc {0} {0} {25} {0}',
        '5': f'rc {25} {0} {0} {0}', '3': f'rc {0} {0} {-25} {0}', '0-4': f'rc {0} {0} {0} {-100}',
        '0-5': f'rc {0} {0} {0} {100}', '0-2': f'rc {0} {100} {0} {0}', '0-3': f'rc {0} {-100} {0} {0}'
    }

    tracker.hands_finder(frame)
    if tracker.results.multi_hand_landmarks is None:
        tello.send_command_without_return('rc 0 0 0 0')
        return

    result: list = []
    for i in range(0, len(gesture_control.hand_tracker.results.multi_hand_landmarks)):
        lms = gesture_control.hand_tracker.position_finder(frame, hand_no=i, normalized=True)

        if lms:
            landmark = np.array(lms).reshape((42,))
            predictions = model.predict(np.array([landmark]))
            predicted = np.argmax(np.squeeze(predictions))
            if np.max(np.squeeze(predictions)) < threshold:
                tello.send_command_without_return('rc 0 0 0 0')
                return
            result.append(int(predicted))
    key = '-'.join(str(e) for e in sorted(result))
    print(key)
    if key == '0':
        if tello.is_flying:
            tello.send_command_without_return('land')
        else:
            tello.send_command_without_return('takeoff')
        return

    if key in commands.keys():
        tello.send_command_without_return(commands[key])

