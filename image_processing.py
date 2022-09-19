import cv2
import numpy as np


def showImg(img: np.ndarray, show_time: int = 0):
    title = 'Image'
    cv2.namedWindow(title)
    cv2.moveWindow(title, 20, 20)
    new_img = resize(img, ())
    cv2.imshow(title, new_img)
    cv2.waitKey(show_time)
    cv2.destroyWindow(title)


def readImg(path: str):
    return cv2.imread(path)


def resize(img: np.ndarray, dim: tuple, factor: float = 1):
    h, w = img.shape[:2]
    if not dim:
        h, w = h * factor, w * factor
        max_h, max_w = 600, 600
        if h > max_h:
            factor = (max_h / h)
        if w * factor > max_w:
            factor = (max_w / w)

        w, h = w * factor, h * factor
    else:
        w, h = dim
    return cv2.resize(img, (int(w), int(h)))


def setToBkgDim(dim_bkg: tuple, dim_fgd: tuple):
    max_w, max_h = dim_bkg
    w, h = dim_fgd
    factor = 1
    if h > max_h:
        factor = ((max_h - 1) / h)
    if w * factor > max_w:
        factor = ((max_w - 1) / w)
    print(factor)
    return int(w * factor), int(h * factor)


def disort(dim: tuple):
    w, h = dim
    pt_x1, pt_y1 = 0, 0
    pt_x2, pt_y2 = int(w * .5), 0
    pt_x3, pt_y3 = 0, int(h * .5)
    pt_x4, pt_y4 = int(w * .5), int(h * .5)
    pts = [[pt_x1, pt_y1], [pt_x2, pt_y2], [pt_x3, pt_y3], [pt_x4, pt_y4]]
    return np.float32(pts)


def createImg(foreground: np.ndarray, background: np.ndarray):

    h_back, w_back = background.shape[:2]
    h_fore, w_fore = foreground.shape[:2]

    w_fore, h_fore = setToBkgDim((w_back, h_back), (w_fore, h_fore))
    foreground = resize(foreground, (w_fore, h_fore))

    pts_src = np.float32([[0, 0], [w_fore, 0], [0, h_fore], [w_fore, h_fore]])

    pts_dst = disort((w_fore, h_fore))
    # result = np.where(warp.sum(axis=-1,keepdims=True)!=0, warp, original)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    new_img = cv2.warpPerspective(
        foreground, matrix, (int(w_back), int(h_back)))

    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # dst = cv2.addWeighted(background, 1, new_img, .5, 0)

    # print(dst)
    # new_h, new_w = new_img.shape[:2]

    return mask
