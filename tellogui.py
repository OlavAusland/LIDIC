import numpy as np
import cv2


def draw_rotation(image: np.ndarray, degree: float):
    """
    Draw rotation on an image (numpy.ndarray)

    :param image: a numpy.ndarray to draw graphics on
    :param degree: rotation of tello drone
    :return:
    """
    center = (image.shape[1] - 50, image.shape[0] - 50)
    cv2.circle(image, center=center, radius=30, color=(255, 0, 255))
    cv2.circle(image, center=(center[0] + int(np.cos(np.deg2rad(degree)) * 30),
                              int(center[1] + np.sin(np.deg2rad(degree)) * 30)),
               thickness=4, radius=4, color=(255, 255, 255))


def draw_roll(image: np.ndarray, degree: float):
    """
    Draw the roll on an image (numpy.ndarray)

    :param image: a numpy.ndarray to draw graphics on
    :param degree: roll of tello drone
    :return:
    """
    left = (image.shape[1] - 80, image.shape[0] - 150)
    cv2.line(image, pt1=left, pt2=[image.shape[1] - 20, image.shape[0] - 150], color=(255, 255, 255))
    cv2.circle(image, center=(image.shape[1] - 50 + int(-(60 * degree / 360)), image.shape[0] - 150), color=(0, 0, 255),
               radius=3, thickness=3)


def draw_pitch(image: np.ndarray, degree: float):
    f"""
    Draw the pitch on an image (numpy.ndarray)
    
    :param image: a image to draw the graphics on
    :type image: {np.ndarray}
    :param degree: str
    :return: 
    """
    bottom = (image.shape[1] - 50, image.shape[0] - 200)
    cv2.line(image, pt1=bottom, pt2=[bottom[0], image.shape[0] - 260], color=(255, 255, 255))
    cv2.circle(image, center=(bottom[0], image.shape[0] - 230 + int(-(60 * degree / 180))), color=(0, 0, 255),
               radius=3, thickness=3)


def draw_velocity(image: np.ndarray, vel: tuple):
    """
    Draw the velocity as a vector on the image

    :param image: a image to draw the graphic on
    :param vel: velocity
    :return:
    """
    cv2.arrowedLine(img=image, pt1=[30, image.shape[0] - 30],
                    pt2=[30 + vel[0], image.shape[0] - 30],
                    color=(0, 0, 255), thickness=2)
    cv2.arrowedLine(img=image, pt1=[30, image.shape[0] - 30],
                    pt2=[30, img.shape[0] - 30 - vel[2]],
                    color=(255, 0, 0), thickness=2)
    cv2.arrowedLine(img=image, pt1=[30, image.shape[0] - 30],
                    pt2=[30 + vel[1], image.shape[0] - 30 - vel[1]],
                    color=(255, 255, 255), thickness=1)
    cv2.circle(image, (30, image.shape[0] - 30), color=(0, 255, 0), thickness=1, radius=2)
