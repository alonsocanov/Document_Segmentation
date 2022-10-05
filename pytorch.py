import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from PIL import Image


def imageTransform(normalize=True):
    # meam for images
    mean = [0.485, 0.456, 0.406]
    # std for images
    std = [0.229, 0.224, 0.225]
    # image resize
    img_resize = 224
    # image crop for neural network (for better training the model)
    img_crop = 224
    # image transforms
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor()])
    return transform


def predict(model_path, img_path, threshold=.7):
    model_param = torch.load(model_path)
    img = Image.open(img_path)
    img = img.convert('RGB')
    model = model_param['model']
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    transform = imageTransform()
    img = transform(img)
    img = img.unsqueeze(0)
    img.to(device)
    output = model(img)
    mask = output['out'].squeeze()
    thresh = torch.nn.Threshold(threshold, 0)
    mask = thresh(mask)
    mask[mask > 0] = 255
    mask = mask.detach().numpy()
    return mask
