from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time
import torch
from segmentation_dataset import SegmentationDataset
import utils
from torchvision import transforms
import torch
import copy
import os
from log import Log
import numpy as np
import sys
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
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


def maskTransform():
    # image resize
    img_resize = 224
    # image crop for neural network (for better training the model)
    img_crop = 224
    # image transforms
    transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_crop),
        transforms.ToTensor()])
    return transform


def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    my_log.message('info', ['Using device:', device])
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    my_log.message('info', ['Fildnames:', fieldnames])

    for epoch in range(1, num_epochs + 1):
        msg = ' '.join(['Epoch', str(epoch), '/', str(num_epochs)])
        my_log.message('info', msg)
        msg = '-'*10
        my_log.message('info', msg)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            my_log.message('info', batchsummary)
            if bpath:

                state = {
                    'model': model,
                    # 'classes': classes,
                    # 'transform': transform,
                    'epoch': epoch,
                    # 'batch_size': batch_size,
                    'device': device
                }

                torch.save(state, bpath)
            msg = [phase, ': Loss:', loss]
            my_log.message('info', msg)
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        # print(batchsummary)
        my_log.message('info', batchsummary)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():

    img_transform = imageTransform()
    mask_transform = maskTransform()
    dataset = SegmentationDataset(
        root, 'dataset/images', 'dataset/masks', img_transform, mask_transform, 'rgb', 'gray')
    train = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                        batch_sampler=None, num_workers=0, drop_last=True)
    # get model
    deepLab = createDeepLabv3()
    # criteria Mean Square Loss
    criterion = torch.nn.MSELoss(reduction='mean')
    my_log.message('info', criterion)
    optimizer = torch.optim.Adam(deepLab.parameters(), lr=0.01)
    num_epochs = 10
    my_log.message('info', ['Number of epochs:', num_epochs])
    lr = 0.0001
    my_log.message('info', ['Learning Rate:', lr])
    save_path = utils.joinPath(root, 'model/doc_segmentation.pth')
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    dataloaders = {'Train': train, 'Test': train}
    img = dataset[0]['image']
    # plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    # plt.show()

    train_model(deepLab, criterion, dataloaders, optimizer,
                metrics, save_path, num_epochs)


root = utils.getParentDir()
log_dir = utils.joinPath(root, 'log')
my_log = Log(log_dir)
my_log.config()
if __name__ == '__main__':
    main()
