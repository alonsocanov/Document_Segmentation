from email.mime import image
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import os
import glob
import sys
from typing import Any, Callable, Optional


class SegmentationDataset(VisionDataset):
    '''
    A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    '''

    def __init__(self, root, img_dir, mask_dir, transforms=None, img_color_mode: str = None, mask_color_mode: str = None, fraction: float = None) -> None:
        super().__init__(root, transforms)
        # check images directories
        if not os.path.isdir(root):
            msg = 'The root directory: ' + root + ' does not exist'
            sys.exit(msg)
        img_path = os.path.join(root, img_dir)
        mask_path = os.path.join(root, mask_dir)
        if not os.path.isdir(img_path):
            msg = 'The images directory: ' + img_path + ' does not exist'
            sys.exit(msg)
        if not os.path.isdir(mask_path):
            msg = 'The mask directory: ' + mask_dir + ' does not exist'
            sys.exit(msg)
        color_modes = ['rgb', 'gray', 'bgr']
        if img_color_mode not in color_modes:
            msg = 'The image color mode  ' + img_color_mode + \
                ' does not exist. It must be ' + str(color_modes)
            sys.exit(msg)
        if mask_color_mode not in color_modes:
            msg = 'The image color mode  ' + mask_color_mode + \
                ' does not exist. It must be ' + str(color_modes)
            sys.exit(msg)

        self.img_color_mode = img_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.img_names = sorted(glob.glob(img_path + '/*.jpg'))
            self.mask_names = sorted(glob.glob(mask_path + '/*.jpg'))
        else:
            # training and test dataset
            pass

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.img_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, 'rb') as image_file, open(mask_path, 'rb') as mask_file:
            img = Image.open(image_file)
            if self.img_color_mode == 'rgb':
                img = img.convert('RGB')
            elif self.img_color_mode == 'bgr':
                img = img.convert('BGR')
            elif self.img_color_mode == 'gray':
                img = img.convert('L')

            mask = Image.open(mask_file)
            if self.mask_color_mode == 'rgb':
                mask = mask.convert('RGB')
            elif self.mask_color_mode == 'bgr':
                mask = mask.convert('BGR')
            elif self.mask_color_mode == 'gray':
                mask = mask.convert('L')

            sample = {'image': img, 'mask': mask}

            if self.transforms:
                sample['image'] = self.transforms(sample['image'])
                sample['mask'] = self.transforms(sample['mask'])
