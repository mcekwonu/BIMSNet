"""Some parts of this code is inspired by LeeJunHyun,
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/evaluation.py, other parts were inspired from a host of
other online sources and youtube videos while other codes was written by Michael Chukwuemeka Ekwonu
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid, save_image, draw_segmentation_masks
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import BIMSDataset


def show_tensor_images(image_tensor, num_images=20, nrow=4, show=True):
    """Visualize images, and display images in a uniform grid:
	
	Parameters:
		image_tensor: image tensor
		num_images: number of images to display
		nrow: images per row to display
		show: flag to display images
	"""
    image_tensor = (image_tensor + 1) / 2
    image_flat = image_tensor.detach().cpu()
    image_grid = make_grid(image_flat[:num_images], nrow=nrow)
    plt.imshow(np.clip(image_grid.permute(1, 2, 0).squeeze(), 0, 1))
    plt.axis("off")
    if show:
        plt.show()


def isempty(path: str) -> None:
    """Checks if the specified directory is empty"""
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path):
            raise Exception(f"{os.path.abspath(path)} is empty. Check the specified directory!")
    os.system("exit")


def compute_max_depth(shape, max_depth=10, verbose=True):
    """Computes the maximum depth of the Network Architecture"""
    shapes = [shape]

    for i in range(1, max_depth):
        if shape % 2 ** i == 0 and shape // 2 ** i > 1:
            shapes.append(shape // 2 ** i)

            if verbose:
                print(f"Level {i}:\t{shape // 2 ** i}")
        else:
            if verbose:
                print(f"Maximum depth: {i - 1}")
            break
    return shapes


def compute_possible_sizes(low, high, depth=8):
    """Computes the possible output shapes"""
    possible_sizes = dict()

    for size in range(low, high + 1):
        sizes = compute_max_depth(size,
                                  max_depth=depth,
                                  verbose=False)
        if len(sizes) == depth:
            possible_sizes[size] = sizes

    return possible_sizes


def get_dataset(image_dir, mask_dir, transform=None):
    if transform is None:
        to_tensor = transforms.ToTensor()
        return BIMSDataset(image_dir, mask_dir, transform=to_tensor)
    else:
        return BIMSDataset(image_dir, mask_dir, transform=transform)


def get_loader(dataset, batch_size=8, num_workers=4, shuffle=False, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, pin_memory=pin_memory)


def composed_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def compute_mean_and_std(loader):
    images, _ = next(iter(loader))
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    if inp_images.size(1) == 1:
        return mean[0], std[0]
    else:
        return mean, std


def save_tensor_images(filename, input_image, target_mask, pred_mask, index, nrow=1, normalize=True):
    """Save grid of input image, target mask and predicted mask during training/prediction.

    Parameter:
    ---------
    filename: str
        Filename to save stack of images
    input_image: tensor
        Input image
    target_mask: tensor
        Groundtruth mask
    pred_mask: tensor
        Predicted mask
    index: int
        Index of images to save
    nrow: int
        Number of rows to stack images, default=1
    normalize: bool
        Normalize image, default=True

    Return:
        None
    """
    images = make_grid(input_image, nrow=nrow, normalize=normalize)
    masks = make_grid(target_mask, nrow=nrow)
    preds = make_grid(pred_mask, nrow=nrow)
    img_grid = torch.cat((images, masks, preds), -1)
    save_image(img_grid, f"{filename}/{index}.png")


def save_segmented_images(filename, pred_mask, index):
    pred_mask = pred_mask[0]
    save_image(pred_mask, f"{filename}/{index}.png")


def save_segmented_images2(filename, pred_mask, index):
    pred_mask = pred_mask[0].cpu().detach().numpy()
    pred_mask = np.squeeze(pred_mask, axis=0)
    pred_mask = pred_mask.astype(np.uint8)
    plt.imsave(f"{filename}/{index}.png", pred_mask, cmap="gray")


def convert_to_time(time):
    """Print the training time in days: hours: minutes: seconds format."""
    days = time // (24 * 60 * 60)
    time %= (24 * 60 * 60)
    hrs = time // (60 * 60)
    time %= (60 * 60)
    mins = time // 60
    time %= 60
    secs = time
    msg = f"Training completed in {days:.0f} days: {hrs:.0f} hours: {mins:.0f} minutes: {secs:.0f} seconds"
    print(msg)
    return msg


def epoch_log(filename, epoch, phase, data):
    with open(filename, "a") as log:
        log.write("{},{},{}\n".format(epoch, phase, data))


def timer_log(filepath, msg):
    with open(filepath, "a") as f:
        f.write(msg)


if __name__ == '__main__':
    compute_max_depth(64)
    print(compute_possible_sizes(2, 64, depth=5))
