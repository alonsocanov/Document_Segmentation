
from log import Log
import utils
import os
import glob

import matplotlib.pyplot as plt

# my libraries
import pytorch as py


def main():
    model_path = utils.joinPath(root, 'model/doc_segmentation.pth')
    img_dir = utils.joinPath(root, 'dataset/images')
    imgs = glob.glob(img_dir + '/*.jpg', recursive=True)
    img_path = imgs[6]
    threshold = 0.6
    mask = py.predict(model_path, img_path, threshold)
    plt.imshow(mask, interpolation='nearest')
    plt.show()


root = utils.getParentDir()
log_dir = utils.joinPath(root, 'log')
my_log = Log(log_dir)
my_log.config()
if __name__ == '__main__':
    main()
