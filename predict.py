
from log import Log
import utils
import glob

# my libraries
import pytorch as py
import image_processing as im


def main():
    model_path = utils.joinPath(root, 'model/doc_segmentation.pth')
    img_dir = utils.joinPath(root, 'dataset/images')
    imgs = glob.glob(img_dir + '/*.jpg', recursive=True)
    img_path = imgs[0]
    threshold = 0.5
    mask = py.predict(model_path, img_path, threshold)
    img = im.readImg(img_path)
    morphed = im.morphImage(img, mask)
    im.showImg(morphed, 0)


root = utils.getParentDir()
log_dir = utils.joinPath(root, 'log')
my_log = Log(log_dir)
my_log.config()
if __name__ == '__main__':
    main()
