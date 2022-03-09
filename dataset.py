import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BIMSDataset(Dataset):
    """Bubble Image Shadowgraph dataset"""

    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return max(len(self.images), len(self.masks))

    def load_image(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        img = Image.open(image_path)
        return img

    def __getitem__(self, index):
        image = self.load_image(index)
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.to_tensor(mask)

