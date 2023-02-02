from djitellopy import Tello
from queue import Queue


def keyboard_controller(tello: Tello, key_queue: Queue):
    """
    Control the drone using the keyboard.
    [Change to take in variable speed]

    :param tello: The tello drone / object to perform the instructions on
    :param key_queue: Key queue to extract key presses
    :return:
    """

    command = {
        ord('w'): f'rc {0} {100} {0} {0}', ord('s'): f'rc {0} {-100} {0} {0}',
        ord('a'): f'rc {-100} {0} {0} {0}', ord('d'): f'rc {100} {0} {0} {0}',
        ord('r'): f'rc {0} {0} {0} {0}', ord('x'): f'rc {0} {0} {0} {0}'}

    key = key_queue.get()
    try:
        if key == ord('x'):
            tello.send_command_without_return(command[key])
            if tello.is_flying: tello.land()
            else: tello.takeoff()
        else:
            tello.send_command_without_return(command[key])
    except Exception as error:
        print(error)

    with key_queue.mutex:
        key_queue.queue.clear()
