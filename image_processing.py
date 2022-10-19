import cv2
import numpy as np
import random


def showImg(img: np.ndarray, show_time: int = 0, title: str = 'Image'):
    cv2.namedWindow(title)
    cv2.moveWindow(title, 20, 20)
    new_img = resize(img)
    cv2.imshow(title, new_img)
    cv2.waitKey(show_time * 1000)
    cv2.destroyWindow(title)


def readImg(path: str):
    return cv2.imread(path)


def resize(img: np.ndarray, dim: tuple = (), factor: float = 1):
    h, w = img.shape[:2]
    if not dim:
        h, w = h * factor, w * factor
        max_h, max_w = 1000, 1000
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
    return int(w * factor), int(h * factor)


def disort(dim: tuple):
    w, h = dim
    max_w_noise = w * 0.1
    max_h_noise = h * 0.1
    rand_w, rand_h = random.random(), random.random()
    noise_w, noise_h = rand_w * max_w_noise, rand_h * max_h_noise
    center_x = (w - max_w_noise) / 2 + noise_w
    center_y = (h - max_h_noise) / 2 + noise_h

    w_r, h_r = dim[0] * .89 / 2, dim[1] * .89 / 2

    pt_x1, pt_y1 = int(center_x - w_r), int(center_y - h_r)
    pt_x2, pt_y2 = int(center_x + w_r), int(center_y - h_r)
    pt_x3, pt_y3 = int(center_x - w_r), int(center_y + h_r)
    pt_x4, pt_y4 = int(center_x + w_r), int(center_y + h_r)
    pts = [[pt_x1, pt_y1], [pt_x2, pt_y2], [pt_x3, pt_y3], [pt_x4, pt_y4]]
    return np.float32(pts)


def saveImg(path: str, img: np.ndarray):
    cv2.imwrite(path, img)


def createImg(foreground: np.ndarray, background: np.ndarray):
    h_back, w_back = background.shape[:2]
    h_back, w_back = 440, 320
    background = resize(background, (w_back, h_back))
    h_fore, w_fore = foreground.shape[:2]
    w_fore, h_fore = setToBkgDim((w_back, h_back), (w_fore, h_fore))
    foreground = resize(foreground, (w_fore, h_fore))

    # Clasification segmented image with class (0 - background, 255 - document)
    # create a black background for segmentation
    # bkg_seg = np.zeros((h_back, w_back, 3), dtype=np.float32)
    # crete document interest for segmentation
    fore_seg = np.ones((h_fore, w_fore, 3), dtype=np.float32) * 255
    # rsc points
    pts_src = np.float32([[0, 0], [w_fore, 0], [0, h_fore], [w_fore, h_fore]])
    # destination points
    pts_dst = disort((w_fore, h_fore))
    # prespective matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # classified image
    class_img = cv2.warpPerspective(
        fore_seg, matrix, (int(w_back), int(h_back)))
    # policy image
    policy_img = cv2.warpPerspective(
        foreground, matrix, (int(w_back), int(h_back)))
    # set images in 8 bit
    background = np.uint8(background)
    class_img = np.uint8(class_img)
    policy_img = np.uint8(policy_img)
    # subtract background
    dataset_img = cv2.subtract(background, class_img)
    dataset_img = cv2.add(dataset_img, policy_img)
    return dataset_img, class_img


def morphImage(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        print('Warning: Unable to morph image')
        return img
    mask[mask == 1] = .5
    mask[mask == 0] = 1
    morph = img.astype(np.float32)
    morph[:, :, 0] = cv2.multiply(morph[:, :, 0], mask)
    morph[:, :, 2] = cv2.multiply(morph[:, :, 2], mask)
    return morph.astype(np.uint8)
