from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time
import torch
from segmentation_dataset import SegmentationDataset
import utils
import torch
import copy
from log import Log
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import argparse
# my libraries
import pytorch as py


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
    start = time.time()
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
                }

                torch.save(state, bpath)
            msg = [phase, ': Loss:', loss]
            my_log.message('info', msg)
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        # print(batchsummary)
        my_log.message('info', batchsummary)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    # saved path
    save_path = utils.joinPath(root, 'model/doc_segmentation.pth')
    dataset_dir = utils.joinPath(root, 'dataset')

    parser = argparse.ArgumentParser(description='Train model for DeepLab')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='If a pretraind model exists use it')
    parser.add_argument('--save', type=bool,
                        default=True, help='Save model')
    parser.add_argument('--num-epochs', type=int, default=40,
                        help='Set number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Set learning rate')
    parser.add_argument('--dataset-path', type=str, default=save_path,
                        help='Path to save model with .pth extension')
    parser.add_argument('--save-path', type=str, default=save_path,
                        help='Path to save model with .pth extension')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size of training')

    args = parser.parse_args()

    lr = args.lr
    my_log.message('info', ['Learning Rate:', lr])
    num_epochs = args.num_epochs
    my_log.message('info', ['Number of epochs:', num_epochs])
    dataset_dir = args.dataset_path
    my_log.message('info', ['Dataset directory:', dataset_dir])

    img_transform = py.imageTransform()
    mask_transform = py.imageTransform(False)
    msg = '------------------- Start training -------------------'
    my_log.message('info', msg)
    dataset = SegmentationDataset(
        root, 'dataset/images', 'dataset/masks', img_transform, mask_transform, 'rgb', 'gray')
    train = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
                                        batch_sampler=None, num_workers=0, drop_last=True)

    save_path = args.save_path
    if utils.fileExists(save_path):
        try:
            msg = 'Loading pretrainded DeepLab model'
            model_param = torch.load(save_path)
            deepLab = model_param['model']
        except:
            # get model
            msg = 'Loading new DeepLab'
            deepLab = createDeepLabv3()
    else:
        # get model
        msg = 'Loading new DeepLab'
        deepLab = createDeepLabv3()
    my_log.message('info', msg)
    # criteria Mean Square Loss
    criterion = torch.nn.MSELoss(reduction='mean')
    my_log.message('info', criterion)
    optimizer = torch.optim.Adam(deepLab.parameters(), lr=lr)

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    dataloaders = {'Train': train, 'Test': train}
    img = dataset[0]['image']

    train_model(deepLab, criterion, dataloaders, optimizer,
                metrics, save_path, num_epochs)


root = utils.getParentDir()
log_dir = utils.joinPath(root, 'log')
my_log = Log(log_dir)
my_log.config()

if __name__ == '__main__':
    main()
