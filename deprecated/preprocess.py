import cv2 as cv
import numpy as np

# 50, 18, 129
# 190, 86, 183


def prep_image(image: np.ndarray) -> np.ndarray:
    image = cv.cvtColor(src=image, dst=image, code=cv.COLOR_RGB2HSV)

    # COLOR THRESHOLD
    lower_thresh = np.array([122, 27, 117], dtype='uint8')
    upper_thresh = np.array([212, 86, 191], dtype='uint8')

    hsv_mask = cv.inRange(src=image, lowerb=lower_thresh, upperb=upper_thresh)

    # SMOOTH
    mask = cv.medianBlur(hsv_mask, 7)
    mask = cv.GaussianBlur(mask, (7, 7), 0)

    rows, cols, channels = image.shape
    image = image[0:rows, 0:cols]

    image = cv.bitwise_or(image, image, mask=mask)
    image = image[0:rows, 0:cols]
    image = cv.cvtColor(src=image, dst=image, code=cv.COLOR_HSV2RGB)

    return image


def main():
    cap = cv.VideoCapture(0)
    _, frame = cap.read()

    cv.namedWindow('feed')
    cv.createTrackbar('H - L', 'feed', 122, 255, lambda x: x)
    cv.createTrackbar('S - L', 'feed', 27, 255, lambda x: x)
    cv.createTrackbar('V - L', 'feed', 117, 255, lambda x: x)
    cv.createTrackbar('H - H', 'feed', 212, 255, lambda x: x)
    cv.createTrackbar('S - H', 'feed', 86, 255, lambda x: x)
    cv.createTrackbar('V - H', 'feed', 191, 255, lambda x: x)
    # MEDIAN BLUR
    cv.createTrackbar('gaussian', 'feed', 6, 255, lambda x: x)
    cv.createTrackbar('median', 'feed', 7, 255, lambda x: x)

    while True:
        _, frame = cap.read()
        default = frame
        frame = cv.cvtColor(src=frame, dst=frame, code=cv.COLOR_RGB2HSV)
        # COLOR THRESHOLD
        lower_thresh = np.array([cv.getTrackbarPos('H - L', 'feed'), cv.getTrackbarPos('S - L', 'feed'), cv.getTrackbarPos('V - L', 'feed')], dtype='uint8')
        upper_thresh = np.array([cv.getTrackbarPos('H - H', 'feed'), cv.getTrackbarPos('S - H', 'feed'), cv.getTrackbarPos('V - H', 'feed')],  dtype='uint8')
        hsv_mask = cv.inRange(src=frame, lowerb=lower_thresh, upperb=upper_thresh)

        # SMOOTH
        val = cv.getTrackbarPos('gaussian', 'feed') if cv.getTrackbarPos('gaussian', 'feed') % 2 == 1 else cv.getTrackbarPos('gaussian', 'feed') + 1
        mask = cv.medianBlur(hsv_mask, cv.getTrackbarPos('median', 'feed') if cv.getTrackbarPos('median', 'feed') % 2 == 1 else 1)
        mask = cv.GaussianBlur(mask, (val, val), 0)

        rows, cols, channels = default.shape
        default = default[0:rows, 0:cols]

        default = cv.bitwise_or(default, default, mask=mask)
        default = default[0:rows, 0:cols]
        default = cv.cvtColor(src=default, dst=default, code=cv.COLOR_HSV2RGB)

        cv.imshow('feed', np.concatenate((hsv_mask, mask), axis=1))
        cv.imshow('result', default)

        if cv.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    main()
