
from log import Log
import utils
import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def imageTransform():
    # meam for images
    mean = [0.485, 0.456, 0.406]
    # std for images
    std = [0.229, 0.224, 0.225]
    # image resize
    img_resize = 224
    # image crop for neural network (for better training the model)
    img_crop = 224
    # image transforms
    transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return transform


def predict(model_path, img_path):
    model_param = torch.load(model_path)
    img = Image.open(img_path)
    img = img.convert('RGB')
    model = model_param['model']
    device = model_param['device']
    model.eval()
    model.to(device)
    transform = imageTransform()
    img = transform(img)
    img = img.unsqueeze(0)
    img.to(device)
    output = model(img)
    mask = output['out'].squeeze()
    thresh = torch.nn.Threshold(.7, 0)
    mask = thresh(mask)
    mask[mask > 0] = 255
    mask = mask.detach().numpy()
    return mask


def main():
    model_path = utils.joinPath(root, 'model/doc_segmentation.pth')
    img_dir = os.path.join(root, 'dataset/images')
    imgs = glob.glob(img_dir + '/*.jpg', recursive=True)
    img_path = imgs[3]
    mask = predict(model_path, img_path)
    plt.imshow(mask, interpolation='nearest')
    plt.show()


root = utils.getParentDir()
log_dir = utils.joinPath(root, 'log')
my_log = Log(log_dir)
my_log.config()
if __name__ == '__main__':
    main()
