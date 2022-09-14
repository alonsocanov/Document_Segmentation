from unittest import result
import cv2
import numpy as np


def showImg(img: np.ndarray, show_time: int = 0):
    title = 'Image'
    cv2.namedWindow(title)
    cv2.moveWindow(title, 20, 20)
    new_img = resize(img)
    cv2.imshow(title, new_img)
    cv2.waitKey(show_time)
    cv2.destroyWindow(title)


def readImg(path: str):
    return cv2.imread(path)


def resize(img: np.ndarray, factor: float = 1):
    h, w = img.shape[:2]
    h, w = h * factor, w * factor
    max_h, max_w = 600, 600
    if h > max_h:
        factor = (max_h / h)
    if w > max_w:
        factor = (max_w / w)

    w, h = w * factor, h * factor

    return cv2.resize(img, (int(w), int(h)))


def createImg(foreground: np.ndarray, background: np.ndarray):

    h_back, w_back = background.shape[:2]
    h_fore, w_fore = foreground.shape[:2]

    pts_src = np.float32([[0, 0], [w_fore, 0], [0, h_fore], [w_fore, h_fore]])
    pts_dst = np.float32(
        [[0, 0], [w_fore*.5, 0], [0, h_fore*.5], [w_fore*.5, h_fore*.5]])

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    new_img = cv2.warpPerspective(
        foreground, matrix, (int(w_fore*.5), int(h_fore*.5)))

    new_h, new_w = new_img.shape[:2]
    print(new_w, new_h, background.shape[:2])
